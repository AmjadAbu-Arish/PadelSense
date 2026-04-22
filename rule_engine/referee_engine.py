class RefereeEngine:
    def __init__(self):
        self.state = "Service"
        self.scores = [0, 0] # Player 1, Player 2
        self.score_mapping = {0: "0", 1: "15", 2: "30", 3: "40", 4: "Ad"}

    def update_state(self, event, mapped_position=None):
        if self.state == "Service":
            if event == "player_hit":
                self.state = "In-Play"
                self.serve_in_progress = True
                return "Valid Service"
            if event == "bounce":
                if getattr(self, 'serve_in_progress', False):
                    self.serve_in_progress = False
                    if mapped_position:
                        mx, my = mapped_position
                        MINI_W, MINI_H, MINI_PAD = 200, 400, 20
                        NET_Y = MINI_PAD + (MINI_H - 2 * MINI_PAD) / 2.0
                        SVC_FRONT_OFFSET = ((MINI_H - 2 * MINI_PAD) / 20.0) * 3.05
                        SVC_BACK_OFFSET = ((MINI_H - 2 * MINI_PAD) / 20.0) * 16.95

                        # In padel, serve must hit the opponent's service box first.
                        if (my > NET_Y and my < SVC_BACK_OFFSET) or (my < NET_Y and my > SVC_FRONT_OFFSET):
                            self.last_bounce_in = True
                            return "IN"
                        else:
                            self.state = "Point-Over"
                            self._increment_score()
                            return "OUT"
            if event == "glass_hit" or event == "fence_hit":
                if getattr(self, 'serve_in_progress', False):
                    # Hit glass before bouncing in service box
                    self.serve_in_progress = False
                    self.state = "Point-Over"
                    self._increment_score()
                    return "OUT"
            if event == "net_contact":
                self.serve_in_progress = False
                self.state = "Point-Over"
                self._increment_score()
                return "NET"
            return "None"

        elif self.state == "In-Play":
            if event == "bounce":
                self.last_bounce_in = False
                # Check if out of bounds based on mapped_position
                if mapped_position:
                    mx, my = mapped_position
                    # Dimensions from mini_court_mapper (MINI_W=200, MINI_H=400, MINI_PAD=20)
                    MINI_W, MINI_H, MINI_PAD = 200, 400, 20
                    if mx < MINI_PAD or mx > MINI_W - MINI_PAD or my < MINI_PAD or my > MINI_H - MINI_PAD:
                        self.state = "Point-Over"
                        self._increment_score()
                        return "OUT"
                self.last_bounce_in = True
                return "IN"
            elif event == "glass_hit" or event == "fence_hit":
                if getattr(self, 'last_bounce_in', False):
                    # Hit turf then glass = IN
                    self.last_bounce_in = False
                    return "IN"
                else:
                    # Hit glass before turf = OUT
                    self.state = "Point-Over"
                    self._increment_score()
                    return "OUT"
            elif event == "net_contact":
                # Assuming net_contact means it didn't cross
                self.state = "Point-Over"
                self._increment_score()
                return "NET"

        elif self.state == "Point-Over":
            # Just a placeholder to wait for next service
            # Could reset state if new service is detected
            if event == "player_hit":
                self.state = "In-Play"
                return "Valid Service"

        return "None"

    def _increment_score(self):
        # Simply increment player 1 score for now to demonstrate UI
        # A real engine would know who hit the ball out
        if self.scores[0] < 4:
            self.scores[0] += 1

    def get_current_score(self):
        s1 = self.score_mapping.get(self.scores[0], "0")
        s2 = self.score_mapping.get(self.scores[1], "0")
        return f"{s1} - {s2}"

def apply_rules(events, ball_positions):
    print("Applying padel rules via RefereeEngine...")
    engine = RefereeEngine()
    decisions = []
    for event, pos in zip(events, ball_positions):
        decision = engine.update_state(event, pos)
        decisions.append(decision)
    return decisions
