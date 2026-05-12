import cv2
import numpy as np
import os
import json
import sys

class ManualCourtSelector:
    def __init__(self, frame, video_hash="default"):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.keypoints = []
        self.video_hash = video_hash
        self.cache_path = f"court_config_{self.video_hash}.json"
        self.window_name = "Select 12 Court Keypoints"
        self.keypoint_names = [
            "Back Left Corner", "Service Line Left", "Net Left", "Service Line Left (Front)",
            "Front Left Corner", "Front Right Corner", "Service Line Right (Front)",
            "Net Right", "Service Line Right", "Back Right Corner",
            "Center Service Line Back", "Center Service Line Front"
        ]

    def select_keypoints(self):
        # Check cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    cached_keypoints = json.load(f)

                if len(cached_keypoints) == 12:
                    # In headless mode, load automatically
                    if not sys.stdin.isatty():
                        print(f"Headless mode: Loading cached keypoints from {self.cache_path}")
                        self.keypoints = [(x, y) for x, y in cached_keypoints]
                        return self.keypoints

                    print(f"Cached keypoints found in {self.cache_path}. Press [S] to skip manual selection and use them, or any other key to re-select.")
                    import select
                    import tty
                    import termios

                    # Read single character from stdin without requiring enter
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        i, o, e = select.select([sys.stdin], [], [], 5) # 5 second timeout
                        if i:
                            ch = sys.stdin.read(1)
                        else:
                            ch = 's' # timeout defaults to skip
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                    if ch.lower() == 's':
                        print("Using cached keypoints.")
                        self.keypoints = [(x, y) for x, y in cached_keypoints]
                        return self.keypoints
            except Exception as e:
                print(f"Failed to load cached keypoints: {e}")

        # Check if GUI support is available
        has_gui = True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
            try:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.keypoints, f)
            except Exception as e:
                print(f"Failed to save keypoints to cache: {e}")

        return self.keypoints

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.keypoints) < 12:
                self.keypoints.append((x, y))
            else:
                print("12 points already selected. Press 'Enter' to confirm or 'u' to undo.")
