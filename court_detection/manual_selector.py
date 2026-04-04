import cv2
import numpy as np

class ManualCourtSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.keypoints = []
        self.window_name = "Select 12 Court Keypoints"
        self.keypoint_names = [
            "Back Left Corner", "Service Line Left", "Net Left", "Service Line Left (Front)",
            "Front Left Corner", "Front Right Corner", "Service Line Right (Front)",
            "Net Right", "Service Line Right", "Back Right Corner",
            "Center Service Line Back", "Center Service Line Front"
        ]

    def select_keypoints(self):
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
        return self.keypoints

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.keypoints) < 12:
                self.keypoints.append((x, y))
            else:
                print("12 points already selected. Press 'Enter' to confirm or 'u' to undo.")
