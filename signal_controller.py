# ============================================================
# SmartFlow - Traffic Signal Controller
# ============================================================

import time
from datetime import datetime


class SignalController:

    """
    Controls one complete signal cycle:

        GREEN
          ↓
        YELLOW
          ↓
        RED
          ↓
        COMPLETED

    This class only handles timing.
    YOLO, LSTM and PSO remain separate.
    """

    def __init__(self):

        self.phase = "WAITING"

        self.running = False

        self.cycle_number = 0

        self.green_time = 0

        self.yellow_time = 0

        self.red_time = 0

        self.phase_started_at = None

        self.phase_ends_at = None


    # ========================================================
    # START CYCLE
    # ========================================================

    def start_cycle(
        self,
        green,
        yellow,
        red,
        cycle_number,
    ):

        self.green_time = int(
            green
        )

        self.yellow_time = int(
            yellow
        )

        self.red_time = int(
            red
        )

        self.cycle_number = int(
            cycle_number
        )

        self.phase = "GREEN"

        self.running = True

        self.phase_started_at = (
            time.time()
        )

        self.phase_ends_at = (
            self.phase_started_at
            + self.green_time
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(self):

        if not self.running:
            return


        now = time.time()


        if now < self.phase_ends_at:
            return


        # ----------------------------------------------------
        # GREEN -> YELLOW
        # ----------------------------------------------------

        if self.phase == "GREEN":

            self.phase = "YELLOW"

            self.phase_started_at = now

            self.phase_ends_at = (
                now + self.yellow_time
            )

            return


        # ----------------------------------------------------
        # YELLOW -> RED
        # ----------------------------------------------------

        if self.phase == "YELLOW":

            self.phase = "RED"

            self.phase_started_at = now

            self.phase_ends_at = (
                now + self.red_time
            )

            return


        # ----------------------------------------------------
        # RED -> COMPLETED
        # ----------------------------------------------------

        if self.phase == "RED":

            self.phase = "COMPLETED"

            self.running = False

            self.phase_started_at = now

            self.phase_ends_at = now


    # ========================================================
    # REMAINING TIME
    # ========================================================

    def remaining(self):

        if not self.running:
            return 0


        remaining = (
            self.phase_ends_at
            - time.time()
        )


        return max(
            0,
            int(
                remaining + 0.999
            ),
        )


    # ========================================================
    # PHASE START
    # ========================================================

    def phase_start_time(self):

        if self.phase_started_at is None:
            return "-"


        return datetime.fromtimestamp(
            self.phase_started_at
        ).strftime(
            "%H:%M:%S"
        )


    # ========================================================
    # PHASE END
    # ========================================================

    def phase_end_time(self):

        if self.phase_ends_at is None:
            return "-"


        return datetime.fromtimestamp(
            self.phase_ends_at
        ).strftime(
            "%H:%M:%S"
        )


    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        return {
            "phase":
                self.phase,

            "running":
                self.running,

            "cycle_number":
                self.cycle_number,

            "green":
                self.green_time,

            "yellow":
                self.yellow_time,

            "red":
                self.red_time,

            "remaining":
                self.remaining(),
        }
