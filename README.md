# OCR Finetune

Small project for OCR dataset processing and model inference.

## Files of interest
- [model.py](model.py) — inference entrypoint; contains [`do_gemma_3n_inference`](model.py).
- [load_ds.py](load_ds.py) — dataset loading and JSON flattening; contains [`flatten_json_to_page_line_words`](load_ds.py).
- [requirements.txt](requirements.txt) — Python dependencies.
- [.env](.env) — environment variables (contains HF_TOKEN in this workspace).
- datasets/ — local dataset folder (large; excluded from VCS by default).

## Quickstart

1. Create a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Populate .env with your HF token (already present in workspace in `.env`).

4. Run:
```sh
python model.py    # runs example inference using the model
python load_ds.py  # example dataset loading / JSON flattening utilities
```

## Notes
- `pdf2image` requires external `poppler` binaries — see project docs.
- The dataset folder may be large; it is excluded from git in the provided .gitignore.