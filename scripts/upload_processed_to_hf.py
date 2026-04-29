"""
One-shot upload of the rolled-up dashboard parquets to Hugging Face under
processed/. Run once after generating new versions of these files.

Run:
    python scripts/upload_processed_to_hf.py
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)

FILES = [
    REPO_DIR / "data" / "processed" / "medicare_partd_2021_2023.parquet",
    REPO_DIR / "data" / "processed" / "medicare_partd_top_providers_by_drug_2021_2023.parquet",
]


def load_token() -> str:
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "rb") as f:
            secrets = tomllib.load(f)
        token = secrets.get("HF_TOKEN")
        if token:
            return token
    raise SystemExit("HF_TOKEN not set in env or .streamlit/secrets.toml")


def main() -> None:
    api = HfApi(token=load_token())
    for path in FILES:
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        size_mb = path.stat().st_size / 1024 / 1024
        target = f"processed/{path.name}"
        print(f"Uploading {path.name} ({size_mb:.1f} MB) → {HF_DATASET_ID}/{target}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            commit_message=f"Add {target}",
        )
        print(f"  done → https://huggingface.co/datasets/{HF_DATASET_ID}/blob/main/{target}")


if __name__ == "__main__":
    main()
