
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# --------------------------------------------------
# 1. Initialize Flask Application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# 2. Configure Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

# --------------------------------------------------
# 3. Model Configuration
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# --------------------------------------------------
# 4. Load Machine Learning Model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed reason=model_file_missing"
        )

        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )

        return model

    except Exception:
        logger.exception("event=model_load_failed")
        raise


# --------------------------------------------------
# 5. Request Logging Middleware
# --------------------------------------------------

@app.before_request
def log_request_start():
    request.request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )

    request.start_time = time.perf_counter()


@app.after_request
def log_request_end(response):
    request_id = getattr(
        request, "request_id", "unknown"
    )

    start_time = getattr(
        request, "start_time", time.perf_counter()
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        (
            "event=http_request request_id=%s "
            "method=%s path=%s status=%s duration_ms=%.2f"
        ),
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------------------------
# 6. Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    logger.info("event=home_endpoint_accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# 7. Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    logger.info("event=health_check status=healthy")

    return jsonify({
        "status": "healthy",
    }), 200


# --------------------------------------------------
# 8. Prediction Endpoint with Error Tracking
# --------------------------------------------------

@app.post("/predict")
def predict():
    request_id = getattr(
        request, "request_id", "unknown"
    )

    # Validate JSON request body
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        logger.warning(
            "event=validation_error request_id=%s "
            "reason=missing_or_invalid_json",
            request_id,
        )

        return jsonify({
            "error": "JSON request body is required",
        }), 400

    # Check required input features
    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error request_id=%s "
            "missing_fields=%s",
            request_id,
            missing_fields,
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        # Prepare input data
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        # Load model and make prediction
        model = load_model()

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS"
            if prediction_code == 1
            else "FAIL"
        )

        # Log successful prediction
        logger.info(
            "event=prediction_success request_id=%s",
            request_id,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except Exception:
        # Log the exception details for debugging
        logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )

        # Return a generic error without exposing internals
        return jsonify({
            "error": "Prediction failed",
            "request_id": request_id,
        }), 500


# --------------------------------------------------
# 9. Start Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "event=application_starting port=5000"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
    )
