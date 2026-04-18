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

    import json

    keypoints = []
    config_path = "court_config.json"

    # Try to load existing configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)

        if video_path in config_data:
            print("\nSaved keypoints found for this video.")
            skip_choice = input("Press [S] to Skip manual selection and use saved points, or any other key to recalibrate: ").strip().lower()
            if skip_choice == 's':
                keypoints = config_data[video_path]
                print("Loaded saved keypoints.")

    if len(keypoints) != 12:
        selector = ManualCourtSelector(first_frame)
        keypoints = selector.select_keypoints()

        if len(keypoints) == 12:
            # Save the newly selected keypoints
            config_data = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    try:
                        config_data = json.load(f)
                    except json.JSONDecodeError:
                        pass

            config_data[video_path] = keypoints
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
            print("Saved new keypoints for future use.")
        elif len(keypoints) == 0:
            print("\nNo keypoints selected. Proceeding without manual court calibration.")
        else:
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

    print("Extracting bounce events for Z-Axis mapping fix...")
    # Get bounces to only map court positions when the ball hits the ground
    bounces = tracker.get_ball_shot_frames(interpolated_positions)

    print("Mapping to mini court...")
    from mini_court.mini_court_mapper import map_to_mini_court
    from mini_court.draw_mini_court import draw_mini_court

    # We map all positions to the mini court but restrict to bounce events
    mini_court_positions = map_to_mini_court(interpolated_positions, keypoints, bounces=bounces)

    print("Tracking players...")
    from player_detector.detector import PlayerTracker
    # Upgrade to YOLO11 for player detection
    player_tracker = PlayerTracker(model_path="yolo11n.pt")

    from mini_court.mini_court_mapper import get_homography
    H_mat = get_homography(keypoints)

    all_tracked_players = []
    for frame in tqdm(frames, desc="Tracking players"):
        detected_players = player_tracker.detect_and_track(frame)
        projected = player_tracker.project_to_mini_court(detected_players, H_mat)
        all_tracked_players.append(projected)

    print("Detecting Events with Bi-LSTM...")
    from event_detector.event_classifier import classify_events
    events = classify_events(lstm_features_df)

    from analysis.speed_analysis import calculate_ball_speed
    speeds = calculate_ball_speed(mini_court_positions, fps=fps)
    if not speeds:
        speeds = [None] * len(frames)

    print("Applying Rules via Referee Engine...")
    from rule_engine.referee_engine import RefereeEngine
    ref_engine = RefereeEngine()
    decisions = []

    for i in range(len(frames)):
        pos = mini_court_positions[i] if mini_court_positions else None
        decision = ref_engine.update_state(events[i], mapped_position=pos)
        decisions.append(decision)

    print("Initializing UI Overlay Drawers...")
    from output_module.overlay_drawer import ScoreboardDrawer, DecisionOverlayDrawer
    scoreboard_drawer = ScoreboardDrawer()
    decision_drawer = DecisionOverlayDrawer()

    last_impact_mini_court_pos = None

    ENABLE_MANUAL_LABELING = False
    if ENABLE_MANUAL_LABELING:
        print("Manual labeling enabled. Press 'b'(bounce), 'g'(glass), 'n'(net), 'o'(out), or space to skip.")
        labels = []
        for i in range(len(frames)):
            frame = frames[i].copy()
            cv2.putText(frame, f"Frame {i}: Press b, g, n, o or space", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            try:
                cv2.imshow("Labeling", frame)
                key = cv2.waitKey(0) & 0xFF
            except cv2.error:
                print("OpenCV GUI not available, skipping manual labeling.")
                key = ord(' ')
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        lstm_features_df['label'] = labels

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print("Saving LSTM features to CSV...")
    lstm_features_df.to_csv(os.path.join("outputs", "lstm_features.csv"), index=False)

    print("Rendering output video...")

    output_path = os.path.join("outputs", "output_video.mp4")
    # Use avc1 for better browser/Streamlit compatibility instead of mp4v
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

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

        # Convert frame to timestamp for logging
        timestamp_sec = i / fps if fps > 0 else 0.0

        # Frame counter display
        cv2.putText(frame, f"Frame: {i}/{frame_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        event_val = events[i]
        decision_val = decisions[i]

        # Display specific event action (e.g., BOUNCE, GLASS)
        if event_val and event_val != "none":
            cv2.putText(frame, f"Action: {event_val.upper()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if decision_val in ["IN", "OUT", "NET"]:
            # Combine decision with event context for better clarity
            full_decision_text = f"{decision_val} ({event_val.upper()})" if event_val and event_val != "none" else decision_val
            decision_drawer.trigger(full_decision_text, frames=int(fps))

            # Replay marker logic: Draw a circle at the impact location
            if x_c is not None and y_c is not None:
                cv2.circle(frame, (int(x_c), int(y_c)), 15, (0, 0, 255), 3)

            # Replay marker logic: save the impact location on the mini-court
            if mini_court_positions and mini_court_positions[i]:
                last_impact_mini_court_pos = mini_court_positions[i]

        csv_data.append({
            'Frame_Index': i,
            'Event_Type': event_val,
            'Decision': decision_val if decision_val in ["IN", "OUT", "NET"] else "",
            'Ball_Speed_kmh': f"{speeds[i]:.2f}" if speeds[i] is not None else ""
        })

        # Apply UI Overlays
        frame = scoreboard_drawer.draw(frame, ref_engine.get_current_score())
        frame = decision_drawer.draw(frame)

        # Draw Mini Court
        if len(keypoints) >= 12 and mini_court_positions: # basic check to ensure mapping exists
            # Pass a trail of up to 15 frames
            trail_start = max(0, i - 15)
            trail = mini_court_positions[trail_start:i+1]

            # Pass impact mark if decision overlay is currently showing
            impact_mark = last_impact_mini_court_pos if decision_drawer.frames_to_show > 0 else None

            frame = draw_mini_court(frame, trail, impact_mark=impact_mark)

        # Draw tracked players
        for p in all_tracked_players[i]:
            px1, py1, px2, py2 = map(int, p['bbox'])
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {p['id']} ({p['team']})", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)

        # Save frame for interactive viewer later
        frames[i] = frame

    out.release()

    print("Saving Match Summary to CSV...")
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join("outputs", "match_summary.csv"), index=False)

    print("Generating Heatmap...")
    from analysis.heatmap_generator import HeatmapGenerator
    heatmap_gen = HeatmapGenerator()
    heatmap_gen.generate(mini_court_positions, output_path=os.path.join("outputs", "heatmap.png"))

    print("Processing complete!")

    # Interactive Frame-by-Frame Viewer
    try:
        cv2.namedWindow("Interactive Viewer")
        print("\n--- Interactive Viewer ---")
        print("Controls: [A] Rewind 1 frame | [D] Forward 1 frame | [W] Play/Pause | [Q] Quit")

        frame_idx = 0
        playing = False

        while True:
            cv2.imshow("Interactive Viewer", frames[frame_idx])

            # Wait for 30ms if playing, otherwise wait indefinitely
            key = cv2.waitKey(30 if playing else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a') and frame_idx > 0:
                frame_idx -= 1
                playing = False
            elif key == ord('d') and frame_idx < len(frames) - 1:
                frame_idx += 1
                playing = False
            elif key == ord('w'):
                playing = not playing

            if playing:
                if frame_idx < len(frames) - 1:
                    frame_idx += 1
                else:
                    playing = False

        cv2.destroyAllWindows()
    except cv2.error:
        print("\nOpenCV GUI not available for Interactive Viewer. Skipping.")

    print("Run 'streamlit run dashboard.py' to view the dashboard.")


if __name__ == "__main__":
    main()