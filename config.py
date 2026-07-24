import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get("HIVE_DB_PATH", os.path.join(BASE_DIR, "hive_data.db"))

API_HOST = os.environ.get("HIVE_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("HIVE_API_PORT", "5000"))
MODEL_PATH = os.path.join(BASE_DIR,"models", "hive_model.h5")


