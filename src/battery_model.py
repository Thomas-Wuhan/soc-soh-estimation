"""
Second-order RC equivalent-circuit model (2-RC ECM) for lithium-ion cells.

The model captures:
  * open-circuit voltage (OCV) as a function of SOC
  * ohmic internal resistance R0
  * two RC branches (R1C1 fast, R2C2 slow) for polarization dynamics

Continuous-time state equations (current I > 0 means discharge):

    dSOC/dt = -eta * I / (3600 * Q_nom)
    dV1/dt  = -V1 / (R1*C1) + I / C1
    dV2/dt  = -V2 / (R2*C2) + I / C2

Terminal voltage:

    V = OCV(SOC) - I*R0 - V1 - V2

This is the plant used both to generate synthetic data and as the prediction
model inside the EKF (a standard "model-based BMS" setup).
"""
from __future__ import annotations

import numpy as np


class BatteryModel:
    """Second-order RC equivalent-circuit model.

    Parameters
    ----------
    capacity_ah : float
        Nominal capacity [Ah].  SOH is defined as capacity_ah / capacity_ah_rated.
    ocv_coeffs : sequence of float, optional
        Polynomial coefficients (highest power first) mapping SOC -> OCV [V].
        The default is a smooth, monotonic NMC-like curve (3.0 V .. 4.2 V).
    r0 : float
        Ohmic internal resistance [Ohm].
    r1, c1 : float
        Fast polarization branch (time constant ~ seconds to tens of seconds).
    r2, c2 : float
        Slow polarization branch (time constant ~ minutes).
    eta : float
        Coulombic efficiency [-].
    """

    def __init__(self,
                 capacity_ah: float = 50.0,
                 ocv_coeffs=None,
                 r0: float = 0.020,
                 r1: float = 0.012, c1: float = 1200.0,
                 r2: float = 0.008, c2: float = 45000.0,
                 eta: float = 1.0) -> None:
        self.capacity_ah = float(capacity_ah)
        self.r0 = float(r0)
        self.r1, self.c1 = float(r1), float(c1)
        self.r2, self.c2 = float(r2), float(c2)
        self.eta = float(eta)

        # OCV(SOC) — monotonic polynomial, verifiable via docv_dsoc() > 0.
        self._ocv_coeffs = np.array(
            ocv_coeffs if ocv_coeffs is not None
            else [0.6, -1.2, 1.4, 0.4, 3.0]
        ).astype(float)
        self._ocv_slope_coeffs = np.polyder(self._ocv_coeffs)

    # ------------------------------------------------------------------ #
    # Open-circuit voltage
    # ------------------------------------------------------------------ #
    def ocv(self, soc):
        """Open-circuit voltage [V] for SOC in [0, 1]."""
        return np.polyval(self._ocv_coeffs, np.clip(soc, 0.0, 1.0))

    def docv_dsoc(self, soc):
        """dV_oc/dSOC — used by the EKF measurement Jacobian."""
        return np.polyval(self._ocv_slope_coeffs, np.clip(soc, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Discretised dynamics
    # ------------------------------------------------------------------ #
    def step(self, current: float, dt: float,
             soc: float, v1: float, v2: float):
        """Advance the model by ``dt`` seconds under a constant current.

        Parameters
        ----------
        current : float
            Applied current [A]; positive = discharge.
        dt : float
            Time step [s].

        Returns
        -------
        (soc_next, v1_next, v2_next, v_terminal)
        """
        tau1, tau2 = self.r1 * self.c1, self.r2 * self.c2
        a1, a2 = np.exp(-dt / tau1), np.exp(-dt / tau2)

        soc_next = soc - self.eta * current * dt / (3600.0 * self.capacity_ah)
        soc_next = float(np.clip(soc_next, 0.0, 1.0))

        v1_next = a1 * v1 + self.r1 * (1.0 - a1) * current
        v2_next = a2 * v2 + self.r2 * (1.0 - a2) * current

        v_term = float(self.ocv(soc_next) - current * self.r0
                       - v1_next - v2_next)
        return soc_next, float(v1_next), float(v2_next), v_term

    # ------------------------------------------------------------------ #
    def simulate(self, current_profile, dt: float, soc0: float = 1.0):
        """Run the plant over a current profile.

        Returns
        -------
        dict with keys ``soc``, ``v``, ``v1``, ``v2``, ``current`` (all arrays).
        """
        current_profile = np.asarray(current_profile, dtype=float)
        n = len(current_profile)
        soc = np.empty(n)
        v = np.empty(n)
        v1a = np.empty(n)
        v2a = np.empty(n)

        s, x1, x2 = float(soc0), 0.0, 0.0
        for k, i_k in enumerate(current_profile):
            s, x1, x2, vk = self.step(i_k, dt, s, x1, x2)
            soc[k], v1a[k], v2a[k], v[k] = s, x1, x2, vk

        return {"soc": soc, "v": v, "v1": v1a, "v2": v2a,
                "current": current_profile}
