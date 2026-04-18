class RefereeEngine:
    def __init__(self):
        self.state = "Service"
        self.scores = [0, 0] # Player 1, Player 2
        self.score_mapping = {0: "0", 1: "15", 2: "30", 3: "40", 4: "Ad"}

    def update_state(self, event, mapped_position=None):
        if self.state == "Service":
            if event == "player_hit":
                self.state = "In-Play"
            elif event == "bounce" and mapped_position:
                # Must hit service box first
                mx, my = mapped_position
                MINI_W, MINI_H, MINI_PAD = 200, 400, 20
                net_y = MINI_H / 2
                svc_front_y = MINI_PAD + (3.05 / 20.0) * (MINI_H - 2 * MINI_PAD)
                svc_back_y = MINI_PAD + (16.95 / 20.0) * (MINI_H - 2 * MINI_PAD)

                # Check if bounce is in service boxes (between net and service lines)
                # This is simplified - real logic would check the specific box based on server
                in_service_box = (mx > MINI_PAD and mx < MINI_W - MINI_PAD) and \
                                 ((my > net_y and my < svc_back_y) or (my < net_y and my > svc_front_y))

                if not in_service_box:
                    self.state = "Point-Over"
                    self._increment_score()
                    return "Fault Service"
            return "Valid Service"

        elif self.state == "In-Play":
            if event == "bounce":
                # Ball hit the turf
                self.last_contact = "turf"

                # Check if out of bounds based on mapped_position
                if mapped_position:
                    mx, my = mapped_position
                    # Dimensions from mini_court_mapper (MINI_W=200, MINI_H=400, MINI_PAD=20)
                    MINI_W, MINI_H, MINI_PAD = 200, 400, 20
                    if mx < MINI_PAD or mx > MINI_W - MINI_PAD or my < MINI_PAD or my > MINI_H - MINI_PAD:
                        self.state = "Point-Over"
                        self._increment_score()
                        return "OUT"
                return "IN"
            elif event == "glass_hit":
                if getattr(self, "last_contact", None) == "turf":
                    # Turf then glass is IN
                    self.last_contact = "glass"
                    return "IN"
                else:
                    # Glass before turf is OUT
                    self.state = "Point-Over"
                    self._increment_score()
                    return "OUT"
            elif event == "net_contact":
                # Assuming net_contact means it didn't cross
                self.state = "Point-Over"
                self._increment_score()
                return "NET"
            elif event == "player_hit":
                # Reset last contact
                self.last_contact = "player"

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
