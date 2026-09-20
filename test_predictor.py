```python
from predictor import predict_next_signal


def main():
    result = predict_next_signal()

    print("\n===== SMARTFLOW PREDICTION =====")

    print(
        f"Predicted Total Vehicles : "
        f"{result['predicted_total_vehicles']:.2f}"
    )

    print(
        f"Predicted Traffic Load  : "
        f"{result['predicted_traffic_load']:.2f}"
    )

    print(
        f"Traffic Class            : "
        f"{result['traffic_class']}"
    )

    print(
        f"Low Threshold            : "
        f"{result['low_threshold']:.2f}"
    )

    print(
        f"High Threshold           : "
        f"{result['high_threshold']:.2f}"
    )

    signal = result["signal"]

    print(
        f"PSO Green Time           : "
        f"{signal['green']:.0f} sec"
    )

    print(
        f"Yellow Time              : "
        f"{signal['yellow']} sec"
    )

    print(
        f"Red Time                 : "
        f"{signal['red']} sec"
    )

    print(
        f"Total Cycle              : "
        f"{signal['cycle']:.0f} sec"
    )

    print("\n===== TEST COMPLETED =====")


if __name__ == "__main__":
    main()
```
