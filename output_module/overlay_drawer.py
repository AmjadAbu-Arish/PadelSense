import cv2

class ScoreboardDrawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.thickness = 2
        self.color = (255, 255, 255)
        self.bg_color = (0, 0, 0)

    def draw(self, frame, score_text):
        h, w = frame.shape[:2]

        # Overlay parameters
        overlay_w, overlay_h = 200, 60
        x1, y1 = w - overlay_w - 20, 20
        x2, y2 = w - 20, 20 + overlay_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.bg_color, -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw text
        text_size = cv2.getTextSize(score_text, self.font, self.font_scale, self.thickness)[0]
        text_x = x1 + (overlay_w - text_size[0]) // 2
        text_y = y1 + (overlay_h + text_size[1]) // 2
        cv2.putText(frame, score_text, (text_x, text_y), self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)

        return frame

class DecisionOverlayDrawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 2.0
        self.thickness = 4
        self.frames_to_show = 0
        self.current_decision = None

    def trigger(self, decision, frames=30):
        if decision in ["IN", "OUT", "NET"]:
            self.current_decision = decision
            self.frames_to_show = frames

    def draw(self, frame):
        if self.frames_to_show > 0 and self.current_decision:
            h, w = frame.shape[:2]
            text = self.current_decision
            color = (0, 255, 0) if text == "IN" else (0, 0, 255) # Green IN, Red OUT/NET

            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2

            # Draw with black outline for better visibility
            cv2.putText(frame, text, (text_x, text_y), self.font, self.font_scale, (0, 0, 0), self.thickness + 4, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), self.font, self.font_scale, color, self.thickness, cv2.LINE_AA)

            self.frames_to_show -= 1

        return frame

def draw_overlays(frame, overlays):
    # Backward compatibility stub
    return frame
