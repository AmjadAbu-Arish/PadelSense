import cv2
import numpy as np

MINI_W = 200
MINI_H = 400
MINI_PAD = 20

def draw_mini_court(frame, trail, impact_mark=None):
    panel = np.zeros((MINI_H, MINI_W, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    p   = MINI_PAD
    w   = MINI_W - 2 * MINI_PAD

    # Needs to match build_mini_court_points logic
    full_w    = 10.0
    full_h    = 20.0
    net_y     = 10.0
    svc_front = 3.05
    svc_back  = 16.95

    def ry(y): return int(p + ((full_h - y) / full_h) * (MINI_H - 2 * MINI_PAD))

    net_y_px = ry(net_y)
    fsl_y_px = ry(svc_front)
    bsl_y_px = ry(svc_back)
    cx    = MINI_W // 2

    cv2.rectangle(panel, (p, p), (p + w, MINI_H - p), (255, 255, 255), 2)
    cv2.line(panel, (p, net_y_px), (p + w, net_y_px), (255, 255, 255), 2)
    cv2.line(panel, (p, fsl_y_px), (p + w, fsl_y_px), (180, 180, 180), 1)
    cv2.line(panel, (p, bsl_y_px), (p + w, bsl_y_px), (180, 180, 180), 1)
    cv2.line(panel, (cx, fsl_y_px), (cx, bsl_y_px), (180, 180, 180), 1)

    # Fading trail
    trail_list = [t for t in trail if t is not None]
    for i, pos in enumerate(trail_list):
        bx, by = int(pos[0]), int(pos[1])
        alpha  = (i + 1) / max(len(trail_list), 1)
        radius = max(2, int(5 * alpha))
        color  = (int(80 * alpha), int(200 * alpha), int(80 * alpha))
        cv2.circle(panel, (bx, by), radius, color, -1)

    # Current ball
    if trail_list:
        bx, by = int(trail_list[-1][0]), int(trail_list[-1][1])
        cv2.circle(panel, (bx, by), 6, (0, 255, 0), -1)
        cv2.circle(panel, (bx, by), 6, (0, 0, 0), 1)

    if impact_mark is not None:
        ix, iy = int(impact_mark[0]), int(impact_mark[1])
        cv2.circle(panel, (ix, iy), 10, (0, 0, 255), 2)
        cv2.circle(panel, (ix, iy), 2, (0, 0, 255), -1)

    # Blend onto main frame
    H, W = frame.shape[:2]

    # Check if frame is large enough
    if H < MINI_H + 20 or W < MINI_W + 20:
        return frame

    y_off = H - MINI_H - 20
    x_off = 20
    roi   = frame[y_off:y_off+MINI_H, x_off:x_off+MINI_W]
    frame[y_off:y_off+MINI_H, x_off:x_off+MINI_W] = cv2.addWeighted(roi, 0.3, panel, 0.7, 0)

    return frame
