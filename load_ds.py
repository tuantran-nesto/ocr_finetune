# from huggingface_hub import hf_hub_download
from datasets import load_dataset
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

document = next(iter(dataset['train']))['pdf']
# print(document)
# pages = convert_from_bytes(pdf_file=document, dpi=300)
# print(f"Number of pages: {len(pages)}")