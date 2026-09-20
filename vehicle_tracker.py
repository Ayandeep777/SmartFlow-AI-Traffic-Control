# ============================================================
# SmartFlow - YOLO + BoT-SORT Vehicle Tracker
# ============================================================

from collections import defaultdict, deque

import cv2


# ============================================================
# COCO VEHICLE CLASS IDs
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# VEHICLE TRACKER
# ============================================================

class VehicleTracker:

    def __init__(self):

        # ----------------------------------------------------
        # IDs that have already been counted
        # ----------------------------------------------------

        self.counted_ids = set()

        # ----------------------------------------------------
        # Session-level cumulative unique counts
        # ----------------------------------------------------

        self.total_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        # ----------------------------------------------------
        # Currently active tracking IDs
        # ----------------------------------------------------

        self.current_ids = set()

        # ----------------------------------------------------
        # ID -> vehicle type
        # ----------------------------------------------------

        self.current_tracks = {}

        # ----------------------------------------------------
        # Track movement history
        # ----------------------------------------------------

        self.track_history = defaultdict(
            lambda: deque(maxlen=20)
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.counted_ids.clear()

        self.total_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        self.current_ids.clear()

        self.current_tracks.clear()

        self.track_history.clear()


    # ========================================================
    # PROCESS TRACKS
    # ========================================================

    def process_tracks(self, boxes):

        current_ids = set()

        current_tracks = {}

        new_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        new_ids = []

        active_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }


        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if boxes is None:

            self.current_ids = set()

            self.current_tracks = {}

            return {
                "boxes": None,

                "new_counts": new_counts,

                "new_ids": new_ids,

                "current_ids": [],

                "active_counts": active_counts,

                "total_counts":
                    self.total_counts.copy(),
            }


        # ----------------------------------------------------
        # Tracking IDs unavailable
        # ----------------------------------------------------

        if boxes.id is None:

            self.current_ids = set()

            self.current_tracks = {}

            return {
                "boxes": boxes,

                "new_counts": new_counts,

                "new_ids": new_ids,

                "current_ids": [],

                "active_counts": active_counts,

                "total_counts":
                    self.total_counts.copy(),
            }


        # ----------------------------------------------------
        # Extract tracking information
        # ----------------------------------------------------

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        class_ids = (
            boxes.cls
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Process every tracked object
        # ----------------------------------------------------

        for (
            box,
            class_id,
            track_id
        ) in zip(
            xyxy,
            class_ids,
            track_ids,
        ):

            class_id = int(class_id)

            track_id = int(track_id)


            # ------------------------------------------------
            # Ignore non-vehicle classes
            # ------------------------------------------------

            if class_id not in VEHICLE_CLASSES:
                continue


            vehicle_type = (
                VEHICLE_CLASSES[class_id]
            )


            # ------------------------------------------------
            # Current active vehicle
            # ------------------------------------------------

            current_ids.add(
                track_id
            )

            current_tracks[
                track_id
            ] = vehicle_type


            # ------------------------------------------------
            # Active vehicle count
            # ------------------------------------------------

            active_counts[
                vehicle_type
            ] += 1


            # ------------------------------------------------
            # Track center
            # ------------------------------------------------

            x1, y1, x2, y2 = box

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            self.track_history[
                track_id
            ].append(
                (
                    center_x,
                    center_y,
                )
            )


            # ------------------------------------------------
            # FIRST TIME THIS TRACK ID IS SEEN
            # ------------------------------------------------

            if (
                track_id
                not in self.counted_ids
            ):

                # Mark ID as counted.
                self.counted_ids.add(
                    track_id
                )

                # Update cumulative unique count.
                self.total_counts[
                    vehicle_type
                ] += 1

                # Record this observation as new.
                new_counts[
                    vehicle_type
                ] += 1

                new_ids.append(
                    track_id
                )


        # ----------------------------------------------------
        # Update current state
        # ----------------------------------------------------

        self.current_ids = (
            current_ids
        )

        self.current_tracks = (
            current_tracks
        )


        # ----------------------------------------------------
        # Return statistics
        # ----------------------------------------------------

        return {

            # Original Ultralytics Boxes object.
            "boxes": boxes,

            # Vehicles seen for first time
            # during this observation.
            "new_counts":
                new_counts,

            "new_ids":
                new_ids,

            # Vehicles currently active.
            "current_ids":
                list(current_ids),

            # Active vehicles by type.
            "active_counts":
                active_counts,

            # Cumulative unique vehicles.
            "total_counts":
                self.total_counts.copy(),
        }


    # ========================================================
    # DRAW TRACKS
    # ========================================================

    def draw_tracks(
        self,
        frame,
        boxes,
        new_ids=None,
    ):

        if boxes is None:
            return frame

        if boxes.id is None:
            return frame

        if new_ids is None:
            new_ids = []


        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        class_ids = (
            boxes.cls
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Draw every tracked vehicle
        # ----------------------------------------------------

        for (
            box,
            class_id,
            track_id
        ) in zip(
            xyxy,
            class_ids,
            track_ids,
        ):

            class_id = int(
                class_id
            )

            track_id = int(
                track_id
            )


            if class_id not in VEHICLE_CLASSES:
                continue


            vehicle_type = (
                VEHICLE_CLASSES[class_id]
            )


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if track_id in new_ids:

                label = (
                    f"{vehicle_type} "
                    f"ID:{track_id} [NEW]"
                )

            else:

                label = (
                    f"{vehicle_type} "
                    f"ID:{track_id}"
                )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )


        return frame
