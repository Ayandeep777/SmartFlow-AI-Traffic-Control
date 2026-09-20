import csv
from datetime import datetime

LIVE_FILE = "SmartFlow_Live_Traffic.csv"


def append_live_row(
    cars,
    motorcycles,
    buses,
    trucks,
    active_vehicles,
    traffic_load,
    new_vehicles,
):
    file_exists = False

    try:
        with open(LIVE_FILE, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cars": cars,
        "motorcycles": motorcycles,
        "buses": buses,
        "trucks": trucks,
        "active_vehicles": active_vehicles,
        "traffic_load": traffic_load,
        "new_vehicles": new_vehicles,
    }

    with open(
        LIVE_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# Simulate 5 seconds of camera data
test_data = [
    (8, 3, 1, 0, 12, 12.5, 12),
    (9, 3, 1, 0, 13, 13.5, 1),
    (10, 2, 1, 1, 14, 17.5, 1),
    (10, 4, 1, 1, 16, 18.5, 2),
    (11, 4, 1, 1, 17, 19.5, 1),
]

for data in test_data:
    append_live_row(*data)

print("Test complete.")
print(f"Created/updated: {LIVE_FILE}")
