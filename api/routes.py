import json
from flask import Blueprint, request, jsonify, current_app, render_template
from buffer.buffer import BUFFER
from models.predict import predict_from_payload
bp = Blueprint("api", __name__)

REQUIRED_FIELDS = [
    "time", "hive_number", "hive_status", "hive_temp", "hive_humidity", "hive_pressure"
]
PREDICTION_CACHE = {"payload": None, "prediction": None}

@bp.route("/api/hive", methods=["POST"])
def receive_hive():
    """
    API endpoint to receive new hive data.
    
    Receives JSON payload, stores the data in the database and
    updates the in-memory buffer.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid or missing JSON"}), 400
    try:
        #  Coerce data types to ensure they are correct
        payload = {
            "time": str(data["time"]),
            "hive_number": int(data["hive_number"]),
            "hive_status": int(data["hive_status"]),
            "hive_temp": float(data["hive_temp"]),
            "hive_humidity": float(data["hive_humidity"]),
            "hive_pressure": float(data["hive_pressure"]),
        }
    except Exception as e:
        return jsonify({"error": "invalid field types", "details": str(e)}), 400

    BUFFER.add(payload)

    ##  FOR DEBUGGING AND INFORMATION
    prediction = None
    try:
        pred_res = predict_from_payload(
            current_payload=payload,
            model_path=current_app.config.get("MODEL_PATH"),
        )
        prediction = pred_res.get("prediction") or pred_res.get("output")
        
        print("\n=== NEW PREDICTION ===")
        print(f"Prediction: {prediction}")
        print("======================\n")
    except Exception as e:
        current_app.logger.exception("Prediction failed")
        print("Prediction error:", str(e))
    return jsonify({
        "status": payload["hive_status"],
        "buffer_size": len(BUFFER.get_all()),
        "prediction": prediction,
        'hive_number': payload["hive_number"],
    }), 201


@bp.route("/")
def dashboard_view():
    """Render the dashboard page located in templates/dashboard.html"""
    return render_template("dashboard.html")


@bp.route("/api/hive/latest", methods=["GET"])
def get_latest():
    """Return the latest payload + prediction for the dashboard to poll."""
    all_items = BUFFER.get_all()
    if not all_items:
        return jsonify({"error": "no data"}), 404

    latest_payload = all_items[-1]
    
    prediction = None
    try:
        pred_res = predict_from_payload(
            current_payload=latest_payload,
            model_path=current_app.config.get("MODEL_PATH"),
        )
        prediction = pred_res.get("prediction") or pred_res.get("output")
    except Exception as e:
        current_app.logger.exception("Prediction failed on latest data")
        print("Prediction error on latest data:", str(e))

    return jsonify({
        "payload": latest_payload,
        "prediction": prediction,
        "buffer_size": len(all_items)
    })

@bp.route("/api/hive/history", methods=["GET"])
def get_history():
    """Return the last N entries (default 20) from the buffer to display a small history table."""
    try:
        n = int(request.args.get("n", 20))
    except Exception:
        n = 20


    all_items = BUFFER.get_all()
    tail = all_items[-n:]
    return jsonify({"history": tail})

# python hive_app/tests/simulator.py
