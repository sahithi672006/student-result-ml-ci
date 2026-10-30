
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from werkzeug.wrappers import Response

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

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "student_result_model.pkl"
)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# --------------------------------------------------
# 4. Define Prometheus Metrics
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction inference attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

# --------------------------------------------------
# 5. Load Machine Learning Model
# --------------------------------------------------

_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)

        logger.error(
            "event=model_load_failed reason=model_file_missing"
        )

        raise FileNotFoundError(
            "student_result_model.pkl was not found."
        )

    try:
        _model = joblib.load(MODEL_PATH)

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )

        return _model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed")
        raise


# --------------------------------------------------
# 6. Request Logging and Metrics Middleware
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

    duration = time.perf_counter() - start_time
    duration_ms = duration * 1000

    path = request.path
    method = request.method
    status = response.status_code

    # Exclude Prometheus scrapes from application traffic metrics.
    if path != "/metrics":
        HTTP_REQUESTS.labels(
            method=method,
            path=path,
            status=str(status),
        ).inc()

        HTTP_LATENCY.labels(
            method=method,
            path=path,
        ).observe(duration)

    log_level = (
        logging.WARNING
        if status >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        (
            "event=http_request request_id=%s "
            "method=%s path=%s status=%s duration_ms=%.2f"
        ),
        request_id,
        method,
        path,
        status,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------------------------
# 7. Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    logger.info("event=home_endpoint_accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# 8. Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    logger.info("event=health_check status=healthy")

    return jsonify({
        "status": "healthy",
    }), 200


# --------------------------------------------------
# 9. Prometheus Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------
# 10. Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    request_id = getattr(
        request, "request_id", "unknown"
    )

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
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS"
            if prediction_code == 1
            else "FAIL"
        )

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s",
            request_id,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )

        return jsonify({
            "error": "Prediction failed",
            "request_id": request_id,
        }), 500


# --------------------------------------------------
# 11. Start Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "event=application_starting port=5000"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
    )
