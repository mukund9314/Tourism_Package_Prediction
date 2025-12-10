import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Hugging Face Login
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# Load Data From Hugging Face
Xtrain_path = "hf://datasets/mukund9314/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/mukund9314/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/mukund9314/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/mukund9314/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# Column Definitions
numeric_features = [
    'Age','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','NumberOfTrips','PitchSatisfactionScore',
    'NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    "TypeofContact","Occupation","Gender","ProductPitched",
    "MaritalStatus","Designation","CityTier","PreferredPropertyStar",
    "Passport","OwnCar"
]

# Class Weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Training with MLflow
with mlflow.start_run():

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= threshold).astype(int)

    # Log metrics
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "test_accuracy": test_report["accuracy"],
    })

    # Save model
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    # Push to Hugging Face
    repo_id = "mukund9314/Tourism-Package-Model"
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
