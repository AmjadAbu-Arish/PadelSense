import cv2
import numpy as np
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path="yolov8n.pt"):
        # We use YOLOv8 to detect class 0 (person)
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Uses ByteTrack to detect and track persons in the frame.
        """
        # tracker="bytetrack.yaml" is built-in with ultralytics
        results = self.model.track(frame, classes=[0], tracker="bytetrack.yaml", persist=True, verbose=False)[0]

        tracked_players = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                tracked_players.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2]
                })

        return tracked_players

    def project_to_mini_court(self, tracked_players, homography_matrix):
        """
        Projects the bottom center of the player's bounding box to the mini court.
        Assigns team based on y-coordinate on the mini court (net is at y=200).
        """
        if homography_matrix is None:
            return []

        projected_players = []
        for player in tracked_players:
            bbox = player["bbox"]
            # Bottom center is usually where the feet are
            feet_x = (bbox[0] + bbox[2]) / 2.0
            feet_y = float(bbox[3])

            pt = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            mini_pt = cv2.perspectiveTransform(pt, homography_matrix)[0][0]

            mx = float(mini_pt[0])
            my = float(mini_pt[1])

            # Net is at y=200 in the mini court (assuming MINI_H=400, net in middle)
            # MINI_H = 400, net_y = 200 roughly.
            # From mini_court_mapper:
            #   ry(10) -> p + (20-10)/20 * 360 = 20 + 0.5*360 = 200
            team = "Team A" if my > 200 else "Team B"

            projected_players.append({
                "id": player["id"],
                "bbox": bbox,
                "mini_court_pos": (mx, my),
                "team": team
            })

        return projected_players
