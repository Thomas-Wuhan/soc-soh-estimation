"""
Extended Kalman Filter (EKF) for State-of-Charge estimation.

State vector      x = [SOC, V1, V2]^T
Control input     u = I  [A]  (positive = discharge)
Measurement       y = V_terminal [V]

The nonlinear measurement equation (OCV as a function of SOC) is linearised
around the current estimate at every step, which is the standard approach in
production BMS firmware.

Notes on the covariance design
------------------------------
* ``Q`` (process noise) encodes how much we trust the model.  A small SOC
  process noise lets the filter trust coulomb counting; too small and the
  filter becomes sluggish and biased.
* ``R`` (measurement noise) reflects current/voltage sensor quality.  Voltage
  is usually the best-conditioned measurement, so R is kept small.
"""
from __future__ import annotations

import numpy as np

try:                                    # supports both `python -m` and direct run
    from .battery_model import BatteryModel
except ImportError:                     # pragma: no cover
    from battery_model import BatteryModel


class SOCEstimatorEKF:
    """EKF-based SOC estimator built on a 2-RC equivalent-circuit model."""

    def __init__(self,
                 model: BatteryModel,
                 dt: float,
                 soc0: float = 1.0,
                 p0=None,
                 q=None,
                 r: float = 4e-4) -> None:
        self.m = model
        self.dt = float(dt)

        # State: [SOC, V1, V2]
        self.x = np.array([soc0, 0.0, 0.0], dtype=float)
        self.P = np.diag(p0 if p0 is not None else [1e-3, 1e-4, 1e-4]).astype(float)
        self.Q = np.diag(q if q is not None else [1e-9, 1e-8, 1e-8]).astype(float)
        self.R = np.array([[float(r)]])

        self.soc_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Pre-computed discrete transition matrices (constant dt)
    # ------------------------------------------------------------------ #
    @property
    def _a1(self) -> float:
        return float(np.exp(-self.dt / (self.m.r1 * self.m.c1)))

    @property
    def _a2(self) -> float:
        return float(np.exp(-self.dt / (self.m.r2 * self.m.c2)))

    def _F(self) -> np.ndarray:
        """Jacobian of the (linear) process model."""
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, self._a1, 0.0],
                         [0.0, 0.0, self._a2]])

    def _B(self) -> np.ndarray:
        """Input matrix (constant, used with u = I)."""
        a1, a2 = self._a1, self._a2
        return np.array([[-self.m.eta * self.dt / (3600.0 * self.m.capacity_ah)],
                         [self.m.r1 * (1.0 - a1)],
                         [self.m.r2 * (1.0 - a2)]])

    def _h(self, x: np.ndarray, current: float) -> float:
        """Measurement function: terminal voltage."""
        soc, v1, v2 = x[0], x[1], x[2]
        return float(self.m.ocv(soc) - current * self.m.r0 - v1 - v2)

    def _H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the measurement function (1x3)."""
        return np.array([[float(self.m.docv_dsoc(x[0])), -1.0, -1.0]])

    # ------------------------------------------------------------------ #
    def step(self, current: float, v_meas: float):
        """One predict + update cycle.

        Returns
        -------
        (soc_estimate, v_predicted)
        """
        # ---- Predict -------------------------------------------------- #
        x_pred = self._F() @ self.x + (self._B() @ np.array([[float(current)]]))[:, 0]
        x_pred[0] = float(np.clip(x_pred[0], 0.0, 1.0))
        P_pred = self._F() @ self.P @ self._F().T + self.Q

        # ---- Update --------------------------------------------------- #
        H = self._H(x_pred)
        y_pred = self._h(x_pred, current)
        S = H @ P_pred @ H.T + self.R               # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)         # Kalman gain

        innovation = float(v_meas) - y_pred
        self.x = x_pred + (K @ np.array([[innovation]]))[:, 0]
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.P = (np.eye(3) - K @ H) @ P_pred

        self.soc_history.append(float(self.x[0]))
        return float(self.x[0]), y_pred

    # ------------------------------------------------------------------ #
    def run(self, current_profile, voltage_meas):
        """Run the filter over a whole profile. Returns estimated SOC array."""
        est = np.empty(len(current_profile))
        for k, (i_k, v_k) in enumerate(zip(current_profile, voltage_meas)):
            est[k], _ = self.step(float(i_k), float(v_k))
        return est
