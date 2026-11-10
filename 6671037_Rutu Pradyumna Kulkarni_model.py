# KPIT Technologies - Apex Internship Stage 2 (Role 1)
# Rutu Pradyumna Kulkarni, B.Tech Mechanical Engineering, COEP Technological University

# Driving risk assessment prototype:
# camera distance -> tracker -> TTC, required deceleration, friction circle -> risk index -> alert level

import math
G = 9.81         # gravity in m/s^2
DT = 0.033       # time step of the 30 Hz camera in seconds
D_MARGIN = 2.0   # standstill buffer in m
LEVELS = ["Level 0 (Nominal)", "Level 1 (Advisory)", "Level 2 (Warning)", "Level 3 (Critical AEB)"]


def clip01(x):
    return max(0.0, min(1.0, x))

class Tracker:
    """Simple 2-state tracker (distance d, closing velocity v), alpha-beta / Kalman-type.
    Not a full Kalman filter: the velocity correction is the position gain / DT,
    damped by a hand-tuned factor of 0.25."""

    def __init__(self, d0):
        self.d, self.v = d0, 0.0            # distance (m), relative velocity (m/s)
        self.var_d, self.var_v = 2.0, 1.0   # variance of d and v
        self.q, self.r = 0.2, 0.8           # process noise, measurement noise due to camera jitter

    def update(self, d_meas):
        # 1. predict: move the distance forward with the current velocity
        self.d += self.v * DT
        self.var_d += self.var_v * DT**2 + self.q
        self.var_v += self.q
        # 2. correct with the measured distance and skip if the camera frame is dropped
        if d_meas is not None:
            gain = self.var_d / (self.var_d + self.r)
            residual = d_meas - self.d
            self.d += gain * residual
            self.v += (gain / DT) * residual * 0.25
            self.var_d *= 1.0 - gain
        return self.d, self.v

def assess(tracker, d_meas, v_ego, yaw_rate, mu, pedestrian=False, d_lane=None, v_lat=0.0):
    """Returns d_est, v_rel, a_max, ttc, a_req, risk (0-100), level (0-3)."""
    d_est, v_rel = tracker.update(d_meas)

    # Friction circle: cornering uses part of the grip, the rest is left for braking
    a_lat = abs(v_ego * yaw_rate)
    a_max = math.sqrt(max(0.5, (mu * G) ** 2 - a_lat ** 2))

    # Longitudinal threat: time to collision and deceleration needed to stop
    d_eff = max(d_est - D_MARGIN, 0.5)
    if v_rel < -0.2:                                   # target is closing in
        ttc = d_eff / abs(v_rel)
        a_req = v_rel ** 2 / (2 * d_eff)
        s_ttc = clip01((3.5 - ttc) / (3.5 - 1.2))      # 0 at TTC >= 3.5 s, 1 at TTC <= 1.2 s
        s_decel = clip01(a_req / a_max)
    else:
        ttc, a_req, s_ttc, s_decel = math.inf, 0.0, 0.0, 0.0

    # Lateral threat (cut-in): time to cross into our lane
    if d_lane is not None and v_lat < -0.1:
        s_tlc = clip01((2.5 - abs(d_lane / v_lat)) / (2.5 - 0.5))   # 0 at TLC >= 2.5 s, 1 at <= 0.5 s
    else:
        s_tlc = 0.0

    # Risk index 0-100, with a 25 % boost for pedestrians
    k_vru = 1.25 if pedestrian else 1.0
    risk = min(100.0, 100 * (0.40 * s_ttc + 0.40 * s_decel + 0.20 * s_tlc) * k_vru)

    # Emergency override
    if (ttc <= 1.2 and v_rel < -1.0) or (a_req >= a_max and a_req > 2.0):
        risk = max(risk, 85.0)

    level = 0 if risk < 20 else 1 if risk < 50 else 2 if risk < 80 else 3
    return d_est, v_rel, a_max, ttc, a_req, risk, level

# name, start gap d0 (m), ego speed (m/s), lead speed (m/s), lead accel (m/s^2), number of frames, friction mu, yaw rate (rad/s), pedestrian?, d_lane (m), v_lat (m/s)
SCENARIOS = [
    ("1 CCRs steady follow",   30.0, 13.89, 13.89,  0.0, 30, 0.85, 0.00, False, None,  0.0),
    ("2 CCRb lead braking",    25.0, 13.89, 13.89, -6.0, 51, 0.85, 0.00, False, None,  0.0),
    ("3 Cut-in on curve",      14.0, 20.00, 17.50,  0.0, 20, 0.65, 0.05, False, 1.0,  -1.2),
    ("4 CPNA pedestrian",      10.0, 11.11,  0.00,  0.0, 15, 0.85, 0.00, True,  0.4, -1.39),
]

def run():
    print(f"{'Scenario':<24}{'d_est':>7}{'v_rel':>8}{'a_max':>7}{'TTC':>6}{'a_req':>7}{'R':>7}  Level")
    for name, d0, v_ego, v_lead, a_lead, frames, mu, yaw, ped, d_lane, v_lat in SCENARIOS:
        d, tracker = d0, None
        for _ in range(frames):
            v_lead = max(0.0, v_lead + a_lead * DT)   # lead vehicle speed
            d += (v_lead - v_ego) * DT                # true gap
            if tracker is None:
                tracker = Tracker(d)
            d_est, v_rel, a_max, ttc, a_req, risk, level = assess(
                tracker, d, v_ego, yaw, mu, ped, d_lane, v_lat)
        ttc_text = "N/A" if ttc == math.inf else f"{ttc:.2f}"
        print(f"{name:<24}{d_est:7.1f}{v_rel:8.2f}{a_max:7.2f}{ttc_text:>6}{a_req:7.2f}{risk:7.1f}  {LEVELS[level]}")

if __name__ == "__main__":
    run()