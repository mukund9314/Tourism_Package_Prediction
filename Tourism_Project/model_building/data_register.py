from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os
from pathlib import Path

repo_id = "mukund9314/Tourism-Package-Prediction"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

data_path = Path("Tourism_Project/data")

# ✅ If data folder missing, exit cleanly (important for CI)
if not data_path.exists():
    print("❌ Data folder not found. Skipping dataset registration.")
    exit(0)

# ✅ Create dataset repo if needed
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("✅ Dataset already exists")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("✅ Dataset repository created")

files = list(data_path.glob("*.csv"))
if not files:
    print("⚠️ No CSV files to upload. Skipping.")
    exit(0)

print(f"✅ Uploading {len(files)} CSV file(s)")
api.upload_folder(
    folder_path=str(data_path),
    repo_id=repo_id,
    repo_type=repo_type,
    allow_patterns=["*.csv"]
)

print("✅ Dataset registration completed")
