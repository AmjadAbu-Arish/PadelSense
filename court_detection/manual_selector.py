import cv2
import numpy as np
import os
import json
import hashlib

class ManualCourtSelector:
    def __init__(self, frame, video_path=None):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.keypoints = []
        self.video_path = video_path
        self.window_name = "Select 12 Court Keypoints"
        self.keypoint_names = [
            "Back Left Corner", "Service Line Left", "Net Left", "Service Line Left (Front)",
            "Front Left Corner", "Front Right Corner", "Service Line Right (Front)",
            "Net Right", "Service Line Right", "Back Right Corner",
            "Center Service Line Back", "Center Service Line Front"
        ]
        self.config_path = None
        if self.video_path:
            # Generate MD5 hash of video path
            hash_md5 = hashlib.md5(self.video_path.encode('utf-8')).hexdigest()
            self.config_path = f"court_config_{hash_md5}.json"

    def _load_cache(self):
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    if "keypoints" in data and len(data["keypoints"]) == 12:
                        return [tuple(kp) for kp in data["keypoints"]]
            except Exception as e:
                print(f"Failed to load cached keypoints: {e}")
        return None

    def _save_cache(self, keypoints):
        if self.config_path and len(keypoints) == 12:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump({"keypoints": keypoints}, f)
                print(f"Keypoints cached to {self.config_path}")
            except Exception as e:
                print(f"Failed to save cached keypoints: {e}")

    def select_keypoints(self):
        cached_keypoints = self._load_cache()
        if cached_keypoints:
            import sys
            if not sys.stdin.isatty():
                print("Headless mode detected: automatically loading cached keypoints.")
                return cached_keypoints
            print("\nCached keypoints found for this video.")
            choice = input("Press [S] to Skip manual selection and use cache, or any other key to re-select: ").strip().lower()
            if choice == 's':
                return cached_keypoints

        # Check if GUI support is available
        has_gui = True
        try:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        except cv2.error as e:
            has_gui = False
            print("\nGUI support not available in OpenCV installation.")
            print("Cannot display keypoint selector.")
            print("\nTo enable GUI, reinstall OpenCV with GUI support:")
            print("  pip uninstall opencv-python -y")
            print("  pip install -U opencv-python")
            return []

        if not has_gui:
            return []

        print("Please click on 12 keypoints of the court in order.")
        print("Press 'u' to undo, 'r' to reset, 'Enter' to confirm, or 'q' to quit.")

        while True:
            display_frame = self.frame.copy()
            for i, kp in enumerate(self.keypoints):
                cv2.circle(display_frame, kp, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (kp[0]+10, kp[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Add general instructions
            cv2.putText(display_frame, "Controls: [U] Undo | [R] Reset | [Enter] Confirm | [Q] Quit",
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if len(self.keypoints) < 12:
                prompt = f"Click to Select Point {len(self.keypoints) + 1}/12:"
                point_name = f"--> {self.keypoint_names[len(self.keypoints)]}"
                cv2.putText(display_frame, prompt, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, point_name, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "All 12 points selected. Press 'Enter' to confirm.", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.keypoints = []
                self.frame = self.original_frame.copy()
            elif key == ord('u') or key == ord('U'):
                if len(self.keypoints) > 0:
                    self.keypoints.pop()
            elif key == 13: # Enter key
                if len(self.keypoints) == 12:
                    print("12 keypoints selected.")
                    break
                else:
                    print(f"Please select all 12 points. Currently selected: {len(self.keypoints)}")

        cv2.destroyWindow(self.window_name)
        if len(self.keypoints) == 12:
            self._save_cache(self.keypoints)
        return self.keypoints

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.keypoints) < 12:
                self.keypoints.append((x, y))
            else:
                print("12 points already selected. Press 'Enter' to confirm or 'u' to undo.")
