#!/usr/bin/env python3
"""Copy checkpoint-300 to syedamaan/qwen3-1.7b-rufus-s1 and attach the model card."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo, snapshot_download, whoami

SRC_REPO = "syedamaan/qwen3-1.7b-rufus-s1-sft-v4-ckpts"
DST = "syedamaan/qwen3-1.7b-rufus-s1"
HERE = Path(__file__).resolve().parent

print("whoami:", whoami()["name"])
src = snapshot_download(SRC_REPO, allow_patterns=["checkpoint-300/**"])
root = Path(src) / "checkpoint-300"
assert (root / "model.safetensors").exists() or list(root.glob("*.safetensors")), root

create_repo(DST, private=True, exist_ok=True, repo_type="model")
api = HfApi()
api.upload_folder(
    folder_path=str(root),
    repo_id=DST,
    repo_type="model",
    ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth"],
)
for p in sorted(HERE.iterdir()):
    if p.suffix.lower() in {".md", ".json", ".png"}:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=DST)
print("shipped", f"https://huggingface.co/{DST}")
