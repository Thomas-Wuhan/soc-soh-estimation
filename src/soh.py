"""
State-of-Health (SOH) estimation.

Two complementary estimators are implemented, mirroring what production BMS
firmware actually ships:

1. **Capacity-based** — Ah-throughput over a known ΔSOC window:

       Q_est = ∫ I dt / ΔSOC            SOH_Q = Q_est / Q_rated

   Robust in operation, but needs the SOC swing to be large enough to keep the
   relative error of ΔSOC small.

2. **Resistance-based** — ohmic resistance from a current step:

       R0_est = ΔV / ΔI                 SOH_R = R0_rated / R0_est

   Reacts fast to ageing and is cheap to compute; noisier because it relies on
   a single transient.
"""
from __future__ import annotations

import numpy as np


def estimate_capacity(current, soc_est, dt: float,
                      soc_lo: float = 0.20, soc_hi: float = 0.90) -> float:
    """Estimate usable capacity [Ah] from an Ah-throughput / ΔSOC pair.

    Parameters
    ----------
    current : array-like
        Current [A], positive = discharge.
    soc_est : array-like
        Estimated SOC [-] for the same samples.
    dt : float
        Sampling period [s].

    Returns
    -------
    float
        Estimated capacity [Ah]. Returns ``nan`` if the SOC swing is too small
        (< 5 %) for a meaningful estimate.
    """
    current = np.asarray(current, dtype=float)
    soc_est = np.asarray(soc_est, dtype=float)

    # Restrict to a well-conditioned SOC window: near 100 % the OCV curve is
    # flat (ΔSOC poorly observable) and near 0 % the cell is clamped — both
    # corrupt the ΔSOC denominator and hence the capacity estimate.
    mask = (soc_est <= soc_hi) & (soc_est >= soc_lo)
    if mask.sum() < 10:
        return float("nan")
    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - 1 - np.argmax(mask[::-1]))

    ah_throughput = float(np.sum(current[i0:i1 + 1]) * dt / 3600.0)   # [Ah]
    dsoc = float(soc_est[i0] - soc_est[i1])

    if abs(dsoc) < 0.05:
        return float("nan")
    return ah_throughput / dsoc


def estimate_resistance(voltage, current, idx: int, window: int = 3) -> float:
    """Estimate ohmic resistance [Ohm] around a current step at ``idx``.

    Uses the instantaneous voltage/current change across the step, which is
    dominated by the ohmic drop (the RC branches cannot move that fast).
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)

    i_before = float(np.mean(current[max(0, idx - window):idx]))
    i_after = float(np.mean(current[idx:idx + window]))
    v_before = float(np.mean(voltage[max(0, idx - window):idx]))
    v_after = float(np.mean(voltage[idx:idx + window]))

    di = i_after - i_before
    if abs(di) < 1e-3:
        return float("nan")
    return abs((v_after - v_before) / di)


def estimate_resistance_avg(voltage, current,
                            n_steps: int = 20, window: int = 1) -> float:
    """Noise-robust R0 estimate: median over the ``n_steps`` largest steps.

    A single current step gives a noisy R0 (one 5 mV sample maps to ~0.25 mOhm
    at 20 A).  Averaging the median over the strongest transients in the
    profile removes most of that variance.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)

    steps = np.abs(np.diff(current))
    order = np.argsort(steps)[::-1][:n_steps]

    estimates = [estimate_resistance(voltage, current, int(i), window)
                 for i in order]
    estimates = [r for r in estimates if not np.isnan(r)]
    if not estimates:
        return float("nan")
    return float(np.median(estimates))


def soh_from_capacity(q_est: float, q_rated: float) -> float:
    """SOH from an estimated capacity."""
    return float(q_est / q_rated)


def soh_from_resistance(r_est: float, r_rated: float) -> float:
    """SOH from an estimated ohmic resistance (inverse relationship)."""
    if not r_est or np.isnan(r_est):
        return float("nan")
    return float(r_rated / r_est)
