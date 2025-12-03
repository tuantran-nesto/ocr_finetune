# from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset
from io import BytesIO
import gzip
from typing import Tuple, List, Dict, Any
from pdf2image import convert_from_bytes
import json


dataset_path = "datasets/lightonai-fc-amf-ocr"
file_name = "fc-amf-train-0000.tar"
json_file_name = "014_2021_01_FC014371187_20210112.json"
json_file_path = f"{dataset_path}/fc-amf-train-0000/{json_file_name}/{json_file_name}"

# hf_hub_download(repo_id="lightonai/fc-amf-ocr", filename=file_name, 
#                 repo_type="dataset",
#                 local_dir=dataset_path,)
dataset = load_dataset('lightonai/fc-amf-ocr', streaming=False, data_files=file_name)
print(next(iter(dataset['train'])).keys())
print(len(dataset['train']))
print(next(iter(dataset['train']))['__key__'])
# Result should be dict_keys(['pdf', 'json.gz', '__key__', '__url__'])
# print(next(iter(dataset['train']))['__url__'])

instruction = "Look at the file and extract all the text from it, following the below format:" +\
"""

[
    {
        "page_idx": int,
        "lines": [
            {
                "text": str
            },
            ...
        ]
    },
    ...
]

"""
# print(next(iter(dataset['train']))['json.gz'])

def flatten_json_to_page_line_words(data):
    """
    Converts the nested OCR JSON structure into a flat format:
    [
        {
            "page_idx": int,
            "lines": [
                {
                    "text": str
                },
                ...
            ]
        },
        ...
    ]
    """
    result = []
    for page in data.get("pages", []):
        page_obj = {"page_idx": page.get("page_idx"), "lines": []}
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = [w.get("value") for w in line.get("words", []) if "value" in w]
                line_text = " ".join(words)
                page_obj["lines"].append({"text": line_text})
        result.append(page_obj)
    return result

# Example usage:

# with open(json_file_path, "r") as f:
#     data = json.load(f)
# flat = flatten_json_to_page_line_words(data)
# print(json.dumps(flat, indent=2))

def convert_to_conversation_for_training(sample, is_image=True):

    if is_image:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "file", "file": sample["pdf"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
        ]
    return {"messages": conversation}

# document = next(iter(dataset['train']))['pdf']
# print(document)
# pages = convert_from_bytes(pdf_file=document, dpi=300)
# print(f"Number of pages: {len(pages)}")
def _load_json_content(json_field):
    """
    Accepts a dataset 'json.gz' field which may be:
    - dict already parsed
    - gzipped bytes
    - str (JSON text) or bytes (JSON text)
    Returns a Python dict (parsed JSON).
    """
    if json_field is None:
        return {}
    # Already a dict
    if isinstance(json_field, dict):
        return json_field
    # gzipped bytes
    if isinstance(json_field, (bytes, bytearray)):
        # try to detect gzip magic
        if len(json_field) >= 2 and json_field[0] == 0x1f and json_field[1] == 0x8b:
            try:
                with gzip.GzipFile(fileobj=BytesIO(json_field)) as gf:
                    raw = gf.read()
                    return json.loads(raw.decode("utf-8"))
            except Exception:
                # fallback: maybe it's plain JSON bytes
                return json.loads(json_field.decode("utf-8"))
        else:
            # plain JSON bytes
            return json.loads(json_field.decode("utf-8"))
    # string
    if isinstance(json_field, str):
        return json.loads(json_field)
    # unknown type
    raise ValueError("Unsupported json.gz field type: %s" % type(json_field))


def _image_to_png_bytes(pil_image) -> bytes:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def prepare_datasets(dataset, train_test_split_ratio: float = 0.8, global_seed: int = 42,
                     pdf_dpi: int = 150) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and test datasets from dataset['train'].
    Returns (train_dataset, test_dataset) each with columns:
      - train_data
      - inference_data
      - key
      - text
    """
    rows: List[Dict[str, Any]] = []
    hf_split = dataset.get("train", dataset)  # accept either a DatasetDict or a Dataset

    for sample in hf_split:
        pdf_field = sample.get("pdf")
        if pdf_field is None:
            continue

        # Convert pdf bytes to PIL images
        try:
            pages = convert_from_bytes(pdf_field, dpi=pdf_dpi)
        except Exception as e:
            # skip problematic PDFs
            print(f"Skipping sample {sample.get('__key__','?')} due to PDF decode error: {e}")
            continue

        # Load and flatten OCR JSON
        try:
            json_obj = _load_json_content(sample.get("json.gz"))
            flat_pages = flatten_json_to_page_line_words(json_obj)  # list of {page_idx, lines:[{text}]}
            # build mapping page_idx -> page_text
            page_text_map = {}
            for p in flat_pages:
                pidx = p.get("page_idx")
                if pidx is None:
                    # fallback: generate incremental when missing
                    continue
                # combine lines into one text string (preserve line breaks)
                lines = [ln.get("text","") for ln in p.get("lines",[])]
                page_text_map[int(pidx)] = "\n".join(lines).strip()
        except Exception as e:
            # If JSON parsing fails, proceed with empty text map
            print(f"Warning: failed to parse json for key {sample.get('__key__','?')}: {e}")
            page_text_map = {}

        base_key = sample.get("__key__") or sample.get("key") or None

        # For each image page, create a row
        for page_index, pil_img in enumerate(pages):
            # Decide text for this page: prefer json page_idx match, else fallback to order
            # Many OCR JSON page_idx are 0-based or 1-based; check both possibilities
            text = ""
            if page_text_map:
                # try matching page_index, then page_index+1
                if page_index in page_text_map:
                    text = page_text_map[page_index]
                elif (page_index + 1) in page_text_map:
                    text = page_text_map[page_index + 1]
                else:
                    # last resort: join all JSON pages text if nothing matched
                    text = page_text_map.get(min(page_text_map.keys()), "")
            # convert image to bytes
            image_bytes = _image_to_png_bytes(pil_img)

            sample_for_conv = {"image": image_bytes, "text": text, "pdf": pdf_field}
            train_data = convert_to_conversation_for_training(sample_for_conv, is_image=True)
            # convert_to_conversation_for_training returns {"messages": [...]}
            inference_data = train_data.get("messages", train_data)  # ensure it's the messages list

            rows.append({
                "train_data": train_data,
                "inference_data": inference_data,
                "key": base_key,
                "text": text
            })

    if not rows:
        raise RuntimeError("No rows prepared from the provided dataset.")

    full_ds = Dataset.from_list(rows)

    # perform deterministic split
    split_result = full_ds.train_test_split(test_size=1.0 - train_test_split_ratio, seed=global_seed)
    train_ds = split_result["train"]
    test_ds = split_result["test"]

    # keep only required columns (already the case)
    keep_cols = ["train_data", "inference_data", "key", "text"]
    # make sure columns exist
    for ds in (train_ds, test_ds):
        for c in keep_cols:
            if c not in ds.column_names:
                ds = ds.add_column(c, [None] * len(ds))
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    return train_ds, test_ds

# Example quick usage:
if __name__ == "__main__":
    # dataset variable is defined earlier in this file in the original code
    train_ds, test_ds = prepare_datasets(dataset, train_test_split_ratio=0.8, global_seed=42)
    print("Prepared train/test sizes:", len(train_ds), len(test_ds))
    # show a sample
    print(train_ds[0])