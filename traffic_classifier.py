# ============================================================
# SmartFlow - Traffic Classification
# ============================================================


def calculate_thresholds(traffic_load_values):
    """
    Calculate LOW and HIGH traffic thresholds.

    The historical traffic-load distribution is divided
    approximately into three sections.
    """

    values = sorted(float(x) for x in traffic_load_values)

    if not values:
        raise ValueError("No traffic-load values available.")

    n = len(values)

    low_index = int(n * 0.3333)
    high_index = int(n * 0.6667)

    low_threshold = values[min(low_index, n - 1)]
    high_threshold = values[min(high_index, n - 1)]

    return {
        "low": low_threshold,
        "high": high_threshold,
    }


def classify_traffic(predicted_load, thresholds):
    """
    Classify predicted traffic load.

    LOW:
        predicted_load <= low threshold

    MEDIUM:
        low < predicted_load <= high

    HIGH:
        predicted_load > high
    """

    predicted_load = float(predicted_load)

    if predicted_load <= thresholds["low"]:
        return "LOW"

    if predicted_load <= thresholds["high"]:
        return "MEDIUM"

    return "HIGH"
