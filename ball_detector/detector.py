import os
import cv2
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class BallTrackerConfig:
    smoothing_alpha: float = 0.5
    max_missed_frames: int = 5
    history_size: int = 10
    max_prediction_step: int = 5
    missed_prediction_boost: float = 1.2
    minimum_change_frames_for_hit: int = 25
    use_tracknet: bool = True

import torch
import torch.nn as nn

class TrackNetV3(nn.Module):
    # A simplified conceptual TrackNet V3 architecture for heatmap prediction
    def __init__(self, in_channels=9, out_channels=256):
        super(TrackNetV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # simplified for placeholder purposes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        return x

class TrackNetFusion:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrackNetV3().to(self.device)
        self.model.eval()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, frames):
        # Process 3 consecutive frames to predict ball heatmaps
        # Here we just simulate the fallback logic if TrackNet cannot find a ball
        # In a real scenario, this would process the sequence of images through the CNN
        return [None] * len(frames)

class BallTracker:
    def __init__(self, config: BallTrackerConfig, model_path: str):
        self.config = config
        self.model = YOLO(model_path)
        if self.config.use_tracknet:
            self.tracknet = TrackNetFusion()
        else:
            self.tracknet = None

    def detect_frames(self, frames: List[np.ndarray], read_from_stub: bool = False, stub_path: Optional[str] = None) -> List[Dict[int, List[float]]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        ball_detections = []
        prev_center = None
        missed_frames = 0

        # Performance optimization: run TrackNet (if used) efficiently
        tracknet_preds = self.tracknet.predict(frames) if self.tracknet else [None] * len(frames)

        # Batching YOLO inferences to improve FPS
        batch_size = 16
        all_results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            # Use FP16 if available for speed
            results = self.model(batch, half=True, verbose=False)
            all_results.extend(results)

        for idx, (frame, results) in enumerate(zip(frames, all_results)):
            boxes = results.boxes

            best_score = -float('inf')
            best_box = None
            best_center = None

            max_dist = self.config.max_prediction_step * (self.config.missed_prediction_boost ** missed_frames)

            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                center_x = (coords[0] + coords[2]) / 2.0
                center_y = (coords[1] + coords[3]) / 2.0

                if prev_center is not None:
                    dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                    dist_score = max(0, 1 - (dist / max(1, max_dist)))
                    score = (dist_score * 0.78) + (conf * 0.22)
                else:
                    score = conf

                if score > best_score:
                    best_score = score
                    best_box = coords
                    best_center = (center_x, center_y)

            # TrackNet fusion logic for handling motion blur and high-speed shots
            tn_pred = tracknet_preds[idx]
            if tn_pred is not None:
                # Normalizing TrackNet's heatmap coordinates to YOLO format [x1, y1, x2, y2]
                tn_x, tn_y, tn_conf = tn_pred
                if tn_conf > 0.5 and (best_score < 0.3 or best_box is None):
                    # Override YOLO with TrackNet prediction
                    box_w, box_h = 10, 10 # Default ball bounding box size from TrackNet center
                    best_box = [tn_x - box_w/2, tn_y - box_h/2, tn_x + box_w/2, tn_y + box_h/2]
                    best_center = (tn_x, tn_y)
                    best_score = tn_conf

            if best_box is not None and (prev_center is None or best_score > 0):
                ball_detections.append({1: best_box})
                prev_center = best_center
                missed_frames = 0
            else:
                ball_detections.append({})
                missed_frames += 1
                if missed_frames > self.config.max_missed_frames:
                    prev_center = None

        if stub_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, List[float]]]) -> List[Dict[int, List[float]]]:
        from filterpy.kalman import KalmanFilter
        # Initialize a Constant Acceleration Kalman Filter
        kf = KalmanFilter(dim_x=6, dim_z=2)
        # State vector: [x, y, vx, vy, ax, ay]
        kf.x = np.zeros(6)

        # State transition matrix (F)
        # Assuming dt = 1 for frames
        dt = 1
        kf.F = np.array([[1, 0, dt,  0, 0.5*dt**2,          0],
                         [0, 1,  0, dt,         0, 0.5*dt**2],
                         [0, 0,  1,  0,        dt,          0],
                         [0, 0,  0,  1,         0,         dt],
                         [0, 0,  0,  0,         1,          0],
                         [0, 0,  0,  0,         0,          1]])

        # Measurement function (H)
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]])

        # Covariance matrices
        kf.P *= 1000.  # Initial state uncertainty
        kf.R = np.eye(2) * 5  # Measurement noise
        from filterpy.common import Q_discrete_white_noise
        # Process noise: assume we can model the unmodeled acceleration
        # block_diag isn't strictly necessary if we just use a simple approx:
        kf.Q = np.eye(6) * 0.1

        interpolated_positions = []
        initialized = False

        # To reconstruct bounding boxes, we'll keep track of the average width and height
        avg_w, avg_h = 0, 0
        w_h_count = 0

        # Run Kalman filter forward
        for frame_idx, frame_data in enumerate(ball_positions):
            kf.predict()

            if 1 in frame_data:
                bbox = frame_data[1]
                x_c = (bbox[0] + bbox[2]) / 2.0
                y_c = (bbox[1] + bbox[3]) / 2.0
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                # Update running average of w and h
                avg_w = (avg_w * w_h_count + w) / (w_h_count + 1)
                avg_h = (avg_h * w_h_count + h) / (w_h_count + 1)
                w_h_count += 1

                z = np.array([x_c, y_c])

                if not initialized:
                    kf.x[:2] = z
                    initialized = True

                kf.update(z)
                # Output smoothed detection
                out_x, out_y = kf.x[0], kf.x[1]
                interpolated_positions.append({1: [out_x - avg_w/2, out_y - avg_h/2, out_x + avg_w/2, out_y + avg_h/2]})
            else:
                if initialized:
                    # Missing detection: use prediction
                    out_x, out_y = kf.x[0], kf.x[1]
                    interpolated_positions.append({1: [out_x - avg_w/2, out_y - avg_h/2, out_x + avg_w/2, out_y + avg_h/2]})
                else:
                    interpolated_positions.append({})

        return interpolated_positions

    def extract_lstm_features(self, interpolated_positions: List[Dict[int, List[float]]], window_size: int = 15) -> pd.DataFrame:
        """
        Extract features for the Deep Learning Event Detector.
        Exports a sequence of the last `window_size` frames of (x, y, dx, dy, velocity, acceleration).
        Returns a DataFrame.
        """
        data = []
        for frame_idx, frame_data in enumerate(interpolated_positions):
            if 1 in frame_data:
                bbox = frame_data[1]
                x_c = (bbox[0] + bbox[2]) / 2.0
                y_c = (bbox[1] + bbox[3]) / 2.0
                data.append({'frame': frame_idx, 'x': x_c, 'y': y_c})
            else:
                data.append({'frame': frame_idx, 'x': None, 'y': None})

        df = pd.DataFrame(data)

        # Convert columns to numeric to avoid diff() TypeError if all are None
        df['x'] = pd.to_numeric(df['x'])
        df['y'] = pd.to_numeric(df['y'])

        # Calculate dx and dy
        df['dx'] = df['x'].diff().fillna(0)
        df['dy'] = df['y'].diff().fillna(0)

        # Calculate velocity (magnitude of dx, dy)
        df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2)

        # Calculate acceleration (diff of velocity)
        df['acceleration'] = df['velocity'].diff().fillna(0)

        # Create windowed features using shift for better performance
        df_out = pd.DataFrame({'frame': df['frame']})

        # Original features
        base_features = ['x', 'y', 'dx', 'dy', 'velocity', 'acceleration']

        # We want history up to window_size. Let's arrange so that
        # suffix _0 is the oldest in window, and _(window_size-1) is the current frame.
        for j in range(window_size):
            shift_amount = window_size - 1 - j
            for col in base_features:
                shifted_col = df[col].shift(shift_amount).fillna(0)
                df_out[f'{col}_{j}'] = shifted_col

        return df_out

    def get_ball_shot_frames(self, ball_positions: List[Dict[int, List[float]]]) -> List[int]:
        data = []
        for frame_idx, frame_data in enumerate(ball_positions):
            if 1 in frame_data:
                bbox = frame_data[1]
                mid_y = (bbox[1] + bbox[3]) / 2.0
                data.append({'frame': frame_idx, 'mid_y': mid_y})
            else:
                data.append({'frame': frame_idx, 'mid_y': None})

        df = pd.DataFrame(data)

        # Calculate rolling mean window=5
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=True).mean()

        # Analyze delta_y
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        hits = []
        direction = 0 # 1 for down, -1 for up
        sustained_frames = 0
        potential_hit_frame = None

        for i in range(1, len(df)):
            delta = df['delta_y'].iloc[i]
            if pd.isna(delta) or delta == 0:
                # If delta is zero or NaN, continue tracking sustained frames
                if direction != 0:
                    sustained_frames += 1
                continue

            current_direction = 1 if delta > 0 else -1

            if current_direction != direction:
                # Direction changed. The inversion happened here.
                # If we had a previous potential hit frame that sustained the required frames, we can't record it here
                # because the logic is to verify the *new* direction is sustained, not the old one.
                # Actually, an inversion is verified if the NEW direction is sustained.

                # So we mark this frame as a potential hit.
                direction = current_direction
                sustained_frames = 1
                potential_hit_frame = df['frame'].iloc[i]
            else:
                sustained_frames += 1

            # If the current direction has been sustained for exactly minimum_change_frames_for_hit,
            # and we have a potential hit frame recorded, then it's a valid hit.
            if sustained_frames == self.config.minimum_change_frames_for_hit and potential_hit_frame is not None:
                hits.append(int(potential_hit_frame))
                potential_hit_frame = None # Reset so we don't count it twice

        return hits

    def draw_bboxes(self, video_frames: List[np.ndarray], ball_positions: List[Dict[int, List[float]]]) -> List[np.ndarray]:
        output_frames = []
        for frame, positions in zip(video_frames, ball_positions):
            out_frame = frame.copy()
            for ball_id, bbox in positions.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(out_frame, f"Ball ID: {ball_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            output_frames.append(out_frame)
        return output_frames

# Stub for compatibility if needed
def detect_ball(frame):
    print("Detecting ball...")
    return []
