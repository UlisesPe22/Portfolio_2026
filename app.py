# hive_app/app.py
from flask import Flask
import config
from api.routes import bp 
from models.database import init_app
import os
def create_app():
    app = Flask(__name__)
    app.config["DATABASE"] = config.DB_PATH
    app.config["MODEL_PATH"] = getattr(config, "MODEL_PATH", None) or os.path.join(app.root_path, "models", "hive_model.tflite")

    # Register API blueprint
    app.register_blueprint(bp)

    init_app(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)
