import cv2
import numpy as np

MINI_W = 200
MINI_H = 400
MINI_PAD = 20

def build_mini_court_points():
    w = MINI_W - 2 * MINI_PAD
    p = MINI_PAD
    full_w    = 10.0
    full_h    = 20.0
    net_y     = 10.0
    svc_front = 3.05
    svc_back  = 16.95

    # Fix the homography orientation by reversing the X-axis map to un-mirror it relative to the camera
    def rx(x): return int(p + ((full_w - x) / full_w) * w)
    def ry(y): return int(p + ((full_h - y) / full_h) * (MINI_H - 2 * MINI_PAD))

    # Our 12 points map to:
    # 0. Back Left Corner: (0, full_h)
    # 1. Service Line Left: (0, svc_back)
    # 2. Net Left: (0, net_y)
    # 3. Service Line Left (Front): (0, svc_front)
    # 4. Front Left Corner: (0, 0)
    # 5. Front Right Corner: (full_w, 0)
    # 6. Service Line Right (Front): (full_w, svc_front)
    # 7. Net Right: (full_w, net_y)
    # 8. Service Line Right: (full_w, svc_back)
    # 9. Back Right Corner: (full_w, full_h)
    # 10. Center Service Line Back: (full_w/2, svc_back)
    # 11. Center Service Line Front: (full_w/2, svc_front)

    return np.array([
        [rx(0),        ry(full_h)],
        [rx(0),        ry(svc_back)],
        [rx(0),        ry(net_y)],
        [rx(0),        ry(svc_front)],
        [rx(0),        ry(0)],
        [rx(full_w),   ry(0)],
        [rx(full_w),   ry(svc_front)],
        [rx(full_w),   ry(net_y)],
        [rx(full_w),   ry(svc_back)],
        [rx(full_w),   ry(full_h)],
        [rx(full_w/2), ry(svc_back)],
        [rx(full_w/2), ry(svc_front)],
    ], dtype=np.float32)

def get_homography(court_keypoints):
    if not court_keypoints or len(court_keypoints) != 12:
        return None
    src_pts = np.array(court_keypoints, dtype=np.float32)
    dst_pts = build_mini_court_points()
    H_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H_mat

def map_to_mini_court(positions, court_keypoints, bounces=None):
    H_mat = get_homography(court_keypoints)
    if H_mat is None:
        print("Cannot map to mini court: invalid keypoints or homography.")
        return []

    print("Mapping positions to mini court...")
    mapped_positions = []

    last_valid_pos = None

    for i, pos_dict in enumerate(positions):
        # Determine if this frame is a bounce frame
        is_bounce = False
        if bounces and i in bounces:
            is_bounce = True

        if 1 in pos_dict:
            bbox = pos_dict[1]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3] # map the bottom of the bounding box

            ball_pt = np.array([[[cx, cy]]], dtype=np.float32)
            mini_pt = cv2.perspectiveTransform(ball_pt, H_mat)[0][0]

            mx = float(mini_pt[0])
            my = float(mini_pt[1])

            # Update position ONLY if it's a bounce OR if we aren't enforcing bounces
            if bounces is None or is_bounce:
                last_valid_pos = (mx, my)

        # Append either the mapped bounce, the last known bounce position, or None
        mapped_positions.append(last_valid_pos)

    return mapped_positions
