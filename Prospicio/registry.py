# save models and open models
import joblib
import os

PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")

def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)
