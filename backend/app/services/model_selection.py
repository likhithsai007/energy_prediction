import os
import json
import joblib
from ..models.train import train_and_evaluate, OUTPUT_DIR

METADATA_FILE = os.path.join(OUTPUT_DIR, "model_metadata.json")

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.metadata = None
        self.models = {}
        self.scaler = None
        self.scaler_kmeans = None
        self.kmeans = None
        self.ensure_models_trained()
        self.load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def ensure_models_trained(self):
        required_files = [
            "linear_regression.joblib",
            "decision_tree.joblib",
            "random_forest.joblib",
            "kmeans.joblib",
            "scaler.joblib",
            "model_metadata.json"
        ]
        all_exist = all(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in required_files)
        if not all_exist:
            print("Trained models not found or incomplete. Training now...")
            train_and_evaluate()

    def load_artifacts(self):
        with open(METADATA_FILE, "r") as f:
            self.metadata = json.load(f)
            
        self.models["Linear Regression"] = joblib.load(os.path.join(OUTPUT_DIR, "linear_regression.joblib"))
        self.models["Decision Tree"] = joblib.load(os.path.join(OUTPUT_DIR, "decision_tree.joblib"))
        self.models["Random Forest"] = joblib.load(os.path.join(OUTPUT_DIR, "random_forest.joblib"))
        self.kmeans = joblib.load(os.path.join(OUTPUT_DIR, "kmeans.joblib"))
        self.scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.joblib"))
        if os.path.exists(os.path.join(OUTPUT_DIR, "scaler_kmeans.joblib")):
            self.scaler_kmeans = joblib.load(os.path.join(OUTPUT_DIR, "scaler_kmeans.joblib"))

    def get_performance(self):
        return {
            "models": self.metadata.get("models_performance", {}),
            "selected_model": self.metadata.get("selected_model", "Random Forest"),
            "selection_reason": self.metadata.get("selection_reason", "")
        }

    def get_patterns(self):
        return self.metadata.get("kmeans_analysis", {})

    def get_feature_importance(self):
        return {
            "selected_model": self.metadata.get("selected_model", "Random Forest"),
            "feature_importance": self.metadata.get("feature_importance", [])
        }

    def get_best_model_and_scaler(self):
        best_name = self.metadata.get("selected_model", "Random Forest")
        model = self.models.get(best_name, self.models.get("Random Forest"))
        return model, self.scaler, self.metadata.get("feature_cols", []), best_name

    def get_kmeans_and_scaler(self):
        return self.kmeans, self.scaler_kmeans, self.metadata.get("cluster_features", []), self.metadata.get("kmeans_analysis", {})
