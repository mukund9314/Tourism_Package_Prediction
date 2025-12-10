from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="Tourism_Project/deployment",
    repo_id="mukund9314/Tourism-Package-Prediction",
    repo_type="space",
    path_in_repo="",
    commit_message="Upload Streamlit app to Hugging Face Space"
)
