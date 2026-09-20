# ============================================================
# SmartFlow - Particle Swarm Optimization
# ============================================================

import random
import math

from config import (
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,

    YELLOW_TIME,
    RED_TIME,

    PSO_PARTICLES,
    PSO_ITERATIONS,

    PSO_INERTIA,
    PSO_COGNITIVE,
    PSO_SOCIAL,

    WAITING_COST_WEIGHT,
    QUEUE_COST_WEIGHT,
    UNDER_SERVICE_PENALTY,
    EXCESSIVE_GREEN_PENALTY,

    SERVICE_RATE,
)


# ============================================================
# COST FUNCTION
# ============================================================

def calculate_cost(
    green_time,
    predicted_load,
):
    """
    Calculate total traffic signal cost.

    Total Cost =
        Waiting Cost
        + Queue Cost
        + Under-Service Penalty
        + Excessive-Green Penalty
    """

    green_time = float(green_time)
    predicted_load = float(predicted_load)

    # --------------------------------------------------------
    # Estimated service during green
    # --------------------------------------------------------

    served_load = (
        SERVICE_RATE
        * green_time
    )

    # --------------------------------------------------------
    # Remaining traffic after green
    # --------------------------------------------------------

    remaining_load = max(
        0.0,
        predicted_load - served_load
    )

    # --------------------------------------------------------
    # Waiting cost
    # --------------------------------------------------------

    waiting_cost = (
        predicted_load
        * max(0.0, green_time / 2.0)
    )

    # --------------------------------------------------------
    # Queue cost
    # --------------------------------------------------------

    queue_cost = (
        remaining_load
        * max(1.0, green_time)
    )

    # --------------------------------------------------------
    # Under-service penalty
    # --------------------------------------------------------

    under_service_penalty = (
        remaining_load
        * UNDER_SERVICE_PENALTY
    )

    # --------------------------------------------------------
    # Excessive green penalty
    # --------------------------------------------------------

    excessive_green = max(
        0.0,
        green_time - 45.0
    )

    excessive_green_penalty = (
        excessive_green
        * EXCESSIVE_GREEN_PENALTY
    )

    # --------------------------------------------------------
    # Total cost
    # --------------------------------------------------------

    total_cost = (
        WAITING_COST_WEIGHT
        * waiting_cost

        + QUEUE_COST_WEIGHT
        * queue_cost

        + under_service_penalty

        + excessive_green_penalty
    )

    return float(total_cost)


# ============================================================
# PSO OPTIMIZATION
# ============================================================

def pso_optimize(predicted_load):
    """
    Optimize green duration using Particle Swarm Optimization.

    Input:
        predicted_load

    Output:
        dictionary containing:

        green
        yellow
        red
        cycle
        cost
    """

    predicted_load = float(
        predicted_load
    )

    # --------------------------------------------------------
    # Particle initialization
    # --------------------------------------------------------

    particles = []

    for _ in range(PSO_PARTICLES):

        position = random.uniform(
            MIN_GREEN_TIME,
            MAX_GREEN_TIME,
        )

        velocity = random.uniform(
            -5.0,
            5.0,
        )

        cost = calculate_cost(
            position,
            predicted_load,
        )

        particles.append(
            {
                "position": position,
                "velocity": velocity,
                "cost": cost,
                "best_position": position,
                "best_cost": cost,
            }
        )


    # --------------------------------------------------------
    # Global best
    # --------------------------------------------------------

    global_best = min(
        particles,
        key=lambda p: p["cost"],
    )

    global_best_position = (
        global_best["position"]
    )

    global_best_cost = (
        global_best["cost"]
    )


    # --------------------------------------------------------
    # PSO iterations
    # --------------------------------------------------------

    for _ in range(PSO_ITERATIONS):

        for particle in particles:

            r1 = random.random()
            r2 = random.random()

            # ----------------------------------------------
            # Velocity update
            # ----------------------------------------------

            particle["velocity"] = (
                PSO_INERTIA
                * particle["velocity"]

                + PSO_COGNITIVE
                * r1
                * (
                    particle["best_position"]
                    - particle["position"]
                )

                + PSO_SOCIAL
                * r2
                * (
                    global_best_position
                    - particle["position"]
                )
            )

            # ----------------------------------------------
            # Position update
            # ----------------------------------------------

            particle["position"] += (
                particle["velocity"]
            )

            # ----------------------------------------------
            # Keep inside allowed green range
            # ----------------------------------------------

            particle["position"] = max(
                MIN_GREEN_TIME,
                min(
                    MAX_GREEN_TIME,
                    particle["position"],
                ),
            )

            # ----------------------------------------------
            # Calculate new cost
            # ----------------------------------------------

            particle["cost"] = calculate_cost(
                particle["position"],
                predicted_load,
            )

            # ----------------------------------------------
            # Personal best
            # ----------------------------------------------

            if (
                particle["cost"]
                < particle["best_cost"]
            ):

                particle["best_cost"] = (
                    particle["cost"]
                )

                particle["best_position"] = (
                    particle["position"]
                )

            # ----------------------------------------------
            # Global best
            # ----------------------------------------------

            if (
                particle["cost"]
                < global_best_cost
            ):

                global_best_cost = (
                    particle["cost"]
                )

                global_best_position = (
                    particle["position"]
                )


    # ========================================================
    # FINAL SIGNAL TIMING
    # ========================================================

    green = int(
        round(global_best_position)
    )

    green = max(
        MIN_GREEN_TIME,
        min(
            MAX_GREEN_TIME,
            green,
        ),
    )

    yellow = int(YELLOW_TIME)
    red = int(RED_TIME)

    cycle = (
        green
        + yellow
        + red
    )


    return {
        "green": green,
        "yellow": yellow,
        "red": red,
        "cycle": cycle,
        "cost": round(
            global_best_cost,
            3,
        ),
    }
