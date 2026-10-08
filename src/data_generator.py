"""
Synthetic duty-cycle generation and sensor-noise injection.

Real BMS validation needs current/voltage data with realistic dynamics:
repetitive load steps (converter, motor, HVAC), regenerative (charge) phases
and rest periods.  This module produces a deterministic, reproducible profile
so results can be regenerated bit-for-bit.
"""
from __future__ import annotations

import numpy as np


def duty_cycle(duration_s: int = 12000,
               dt: float = 1.0,
               base_load_a: float = 12.0,
               pulse_a: float = 18.0,
               regen_a: float = -10.0,
               seed: int = 42) -> np.ndarray:
    """Repetitive pulse-with-rest profile (simplified drive/duty cycle).

    The profile alternates between:

    * **load**    — base load plus a stochastic superimposed pulse
    * **regen**   — a short charging (negative current) phase
    * **rest**    — near-zero current (relaxation)

    Parameters
    ----------
    duration_s : int
        Total duration [s].
    dt : float
        Sampling period [s].
    base_load_a : float
        Average discharge current [A].
    pulse_a : float
        Amplitude of the superimposed stochastic pulse [A].
    regen_a : float
        Regenerative (charging) current [A].
    seed : int
        RNG seed, for reproducibility.

    Returns
    -------
    np.ndarray
        Current profile [A], positive = discharge.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    current = np.empty(n)

    # Cycle structure: 120 s load, 20 s regen, 30 s rest  (170 s period)
    for k in range(n):
        phase = (k * dt) % 170.0
        if phase < 120.0:
            current[k] = base_load_a + pulse_a * abs(rng.normal(0.0, 0.6))
        elif phase < 140.0:
            current[k] = regen_a + 2.0 * rng.normal(0.0, 0.4)
        else:
            current[k] = 0.5 * rng.normal(0.0, 0.5)

    return current


def add_sensor_noise(signal,
                     sigma: float,
                     seed: int = 7) -> np.ndarray:
    """Add zero-mean Gaussian measurement noise to a signal."""
    rng = np.random.default_rng(seed)
    return np.asarray(signal, dtype=float) + sigma * rng.normal(size=len(signal))


def aged_cell(base: "BatteryModel",  # noqa: F821  (type hint kept lenient)
              soh: float = 0.8):
    """Return a copy of ``base`` with capacity/resistance degraded to ``soh``.

    A crude but physically sensible ageing model:

    * capacity fades linearly with SOH
    * ohmic resistance grows as the cell ages (inverse-SOH relation,
      a widely used first-order approximation)
    """
    import copy

    cell = copy.deepcopy(base)
    cell.capacity_ah = base.capacity_ah * soh
    cell.r0 = base.r0 / soh
    cell.r1 = base.r1 / soh
    cell.r2 = base.r2 / soh
    return cell
