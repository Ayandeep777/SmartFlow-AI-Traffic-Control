# ============================================================
# SmartFlow - Traffic Feature Calculations
# ============================================================

from config import (
    MOTORCYCLE_WEIGHT,
    CAR_WEIGHT,
    BUS_WEIGHT,
    TRUCK_WEIGHT,
)


def calculate_traffic_load(
    cars,
    motorcycles,
    buses,
    trucks,
):
    """
    Calculate Weighted Traffic Load.

    Formula:

        Load =
            cars * 1.0
            + motorcycles * 0.5
            + buses * 3.0
            + trucks * 3.5
    """

    load = (
        cars * CAR_WEIGHT
        + motorcycles * MOTORCYCLE_WEIGHT
        + buses * BUS_WEIGHT
        + trucks * TRUCK_WEIGHT
    )

    return float(load)
