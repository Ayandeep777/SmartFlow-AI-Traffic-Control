from predictor import predict_from_live
from config import TIME_STEPS

# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------
# These are example traffic-load values.
# We only need 30 values to test the trained LSTM.

test_values = [
    8, 10, 12, 11, 14,
    15, 13, 16, 18, 17,
    20, 19, 21, 22, 20,
    23, 25, 24, 26, 28,
    27, 29, 30, 28, 31,
    32, 30, 33, 35, 34
]

# ------------------------------------------------------------
# CHECK DATA LENGTH
# ------------------------------------------------------------

if len(test_values) < TIME_STEPS:
    raise ValueError(
        f"Need at least {TIME_STEPS} values."
    )

# ------------------------------------------------------------
# RUN LSTM + PSO
# ------------------------------------------------------------

result = predict_from_live(
    test_values
)

# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

print("\n====================================")
print("SMARTFLOW PREDICTION TEST")
print("====================================")

print(
    f"Predicted Traffic Load : "
    f"{result['prediction']:.2f}"
)

print(
    f"Traffic Level          : "
    f"{result['level']}"
)

print(
    f"PSO Green Time         : "
    f"{result['green']} sec"
)

print(
    f"Yellow Time            : "
    f"{result['yellow']} sec"
)

print(
    f"Red Time               : "
    f"{result['red']} sec"
)

print(
    f"Total Cycle            : "
    f"{result['cycle']} sec"
)

print("====================================")
