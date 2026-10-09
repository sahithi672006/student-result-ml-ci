
import json
import sys

MINIMUM_ACCURACY = 0.99

print("Reading model evaluation metrics...")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]

    print("Model Accuracy :", round(accuracy, 4))
    print("Required Accuracy:", MINIMUM_ACCURACY)

    if accuracy < MINIMUM_ACCURACY:
        print("QUALITY GATE FAILED")
        print("Model performance is below the required threshold.")
        sys.exit(1)

    print("QUALITY GATE PASSED")
    print("Model performance satisfies the required threshold.")
    sys.exit(0)

except FileNotFoundError:
    print("ERROR: metrics.json was not found.")
    sys.exit(1)

except (json.JSONDecodeError, KeyError):
    print("ERROR: Invalid metrics.json or missing accuracy.")
    sys.exit(1)
