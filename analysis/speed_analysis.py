import numpy as np
import pandas as pd

def calculate_ball_speed(mapped_positions, fps=30.0):
    """
    Calculates the speed of the ball in km/h based on positions mapped to the mini-court.
    The mini-court is assumed to represent a real 10m x 20m padel court.
    MINI_W = 200, MINI_H = 400, MINI_PAD = 20.
    The playable area in the mini-court is 160x360 pixels.
    This corresponds to 10m x 20m.
    """
    print("Calculating ball speed...")
    # Scale from mini-court pixels to real-world meters
    # width: 160 pixels = 10 meters => 1 pixel = 10/160 meters = 0.0625 meters
    # height: 360 pixels = 20 meters => 1 pixel = 20/360 meters = 0.0555... meters
    # For simplicity, we can use an average scaling factor or exact x/y scaling
    px_to_m_x = 10.0 / 160.0
    px_to_m_y = 20.0 / 360.0

    speeds = []

    for i in range(len(mapped_positions)):
        if i == 0 or mapped_positions[i] is None or mapped_positions[i-1] is None:
            speeds.append(0.0)
            continue

        curr_x, curr_y = mapped_positions[i]
        prev_x, prev_y = mapped_positions[i-1]

        # Calculate real-world distance in meters
        dx_m = (curr_x - prev_x) * px_to_m_x
        dy_m = (curr_y - prev_y) * px_to_m_y

        dist_m = np.sqrt(dx_m**2 + dy_m**2)

        # Speed in meters per second
        speed_mps = dist_m * fps

        # Speed in km/h
        speed_kmh = speed_mps * 3.6
        speeds.append(speed_kmh)

    return speeds
