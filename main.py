from input_handler import (
    DisplayConfig,
    InputHandlerRuntime,
    PreprocessingConfig,
)


import os

def choose_video_file():
    input_dir = "input_videos"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not videos:
        print(f"\nNo videos found in '{input_dir}/'. Please place a video there.")
        print("Alternatively, paste the full video path manually.\n")
        path = input("Video path: ").strip().strip('"').strip("'")
        return path

    print("\nAvailable videos in 'input_videos/':")
    for i, vid in enumerate(videos):
        print(f"[{i+1}] {vid}")

    print(f"\nSelect a video [1-{len(videos)}] or paste a full path:")
    choice = input("Choice/Path: ").strip().strip('"').strip("'")

    if choice.isdigit() and 1 <= int(choice) <= len(videos):
        return os.path.join(input_dir, videos[int(choice)-1])
    else:
        return choice


import cv2
import pandas as pd
from tqdm import tqdm
from court_detection.manual_selector import ManualCourtSelector
from ball_detector.detector import BallTracker, BallTrackerConfig

def main():
    video_path = choose_video_file()

    if not video_path:
        print("No video selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        return

    selector = ManualCourtSelector(first_frame)
    keypoints = selector.select_keypoints()

    if len(keypoints) == 0:
        print("\nNo keypoints selected. Proceeding without manual court calibration.")
    elif len(keypoints) != 12:
        print(f"\n{len(keypoints)} keypoints were selected (expected 12). Proceeding with available keypoints.")

    # Reload video to read all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Reading video frames...")
    frames = []
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print("Initializing BallTracker...")
    config = BallTrackerConfig()
    model_path = "models/ball_detector/ball_detector.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Using default YOLOv8n model instead.")
        model_path = "yolov8n.pt"
    tracker = BallTracker(config, model_path)

    print("Running detection...")
    raw_detections = tracker.detect_frames(frames, read_from_stub=False)

    print("Interpolating ball positions...")
    interpolated_positions = tracker.interpolate_ball_positions(raw_detections)

    print("Extracting LSTM features...")
    lstm_features_df = tracker.extract_lstm_features(interpolated_positions)

    print("Mapping to mini court...")
    from mini_court.mini_court_mapper import map_to_mini_court
    from mini_court.draw_mini_court import draw_mini_court

    # We map all positions to the mini court
    mini_court_positions = map_to_mini_court(interpolated_positions, keypoints)

    ENABLE_MANUAL_LABELING = False
    if ENABLE_MANUAL_LABELING:
        print("Manual labeling enabled. Press 'b'(bounce), 'g'(glass), 'n'(net), 'o'(out), or space to skip.")
        labels = []
        for i in range(len(frames)):
            frame = frames[i].copy()
            cv2.putText(frame, f"Frame {i}: Press b, g, n, o or space", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Labeling", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('b'):
                labels.append("bounce")
            elif key == ord('g'):
                labels.append("glass")
            elif key == ord('n'):
                labels.append("net")
            elif key == ord('o'):
                labels.append("out")
            else:
                labels.append("none")
        cv2.destroyAllWindows()
        lstm_features_df['label'] = labels

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print("Saving LSTM features to CSV...")
    lstm_features_df.to_csv(os.path.join("outputs", "lstm_features.csv"), index=False)

    print("Rendering output video...")

    output_path = os.path.join("outputs", "output_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_data = []

    for i in tqdm(range(len(frames))):
        frame = frames[i].copy()

        # Draw Keypoints
        for i_kp, kp in enumerate(keypoints):
            cv2.circle(frame, kp, 5, (0, 0, 255), -1)

        # Draw BBox
        pos = interpolated_positions[i]
        conf = 1.0
        x_c, y_c = None, None

        if 1 in pos:
            bbox = pos[1]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Ball ID: 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            x_c = (x1 + x2) / 2.0
            y_c = (y1 + y2) / 2.0

            # Check if this was an original detection or interpolated by checking raw_detections
            if 1 in raw_detections[i]:
                conf = 1.0
            else:
                conf = 0.5

        near_line = False
        low_confidence_warning = False

        # Check distance to court lines if mapped
        if len(keypoints) >= 12 and mini_court_positions and mini_court_positions[i] is not None:
            mx, my = mini_court_positions[i]
            # Dimensions from mini_court_mapper (MINI_W=200, MINI_H=400, MINI_PAD=20)
            # The actual court boundaries on the mini court are between MINI_PAD and MINI_W/H - MINI_PAD
            MINI_W, MINI_H, MINI_PAD = 200, 400, 20

            # Simple check if the ball is near any boundary line on the mini court
            # E.g., near x = 20, x = 180, y = 20, y = 380, y = 200 (net)
            dist_left = abs(mx - MINI_PAD)
            dist_right = abs(mx - (MINI_W - MINI_PAD))
            dist_top = abs(my - MINI_PAD)
            dist_bottom = abs(my - (MINI_H - MINI_PAD))

            # Threshold in mini-court pixels
            NEAR_LINE_THRESHOLD = 5.0

            if min(dist_left, dist_right, dist_top, dist_bottom) < NEAR_LINE_THRESHOLD:
                near_line = True
                if conf == 0.5: # Indicates it was predicted/interpolated by checking our dummy value
                    low_confidence_warning = True

        csv_data.append({
            'Frame': i,
            'X': x_c,
            'Y': y_c,
            'Confidence': conf if x_c is not None else None,
            'Near_Line': near_line,
            'Low_Confidence_Warning': low_confidence_warning
        })

        # Draw Mini Court
        if len(keypoints) >= 12 and mini_court_positions: # basic check to ensure mapping exists
            # Pass a trail of up to 15 frames
            trail_start = max(0, i - 15)
            trail = mini_court_positions[trail_start:i+1]
            frame = draw_mini_court(frame, trail)

        out.write(frame)

    out.release()

    print("Saving ball coordinates to CSV...")
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join("outputs", "ball_coordinates.csv"), index=False)

    print("Processing complete!")


if __name__ == "__main__":
    main()