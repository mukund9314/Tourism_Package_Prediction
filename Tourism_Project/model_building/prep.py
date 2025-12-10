import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

DATASET_REPO = "mukund9314/Tourism-Package-Prediction"

# ✅ Download dataset
csv_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset"
)

df = pd.read_csv(csv_path)

# ✅ Basic preprocessing
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Save prepared data locally for training step
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Data preparation completed successfully")
