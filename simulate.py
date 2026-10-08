"""
Main entry point — SOC estimation with an EKF, plus SOH estimation.

Run:
    python simulate.py

Outputs:
    results/soc_estimation.png    — SOC tracking, error and voltage residuals
    results/soh_estimation.png    — SOH estimates vs ground truth
    results/metrics.txt           — numeric summary

Engineering notes
-----------------
The EKF deliberately runs on a *mismatched* model (parameters a few percent
off the plant, as in any real BMS where the cell model is identified offline
and drifts with temperature and ageing).  Running the filter on the exact
plant would produce unrealistically tiny errors and would not demonstrate
robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from battery_model import BatteryModel                     # noqa: E402
from data_generator import duty_cycle, add_sensor_noise, aged_cell  # noqa: E402
from ekf import SOCEstimatorEKF                            # noqa: E402
from soh import (estimate_capacity, estimate_resistance_avg,   # noqa: E402
                 soh_from_capacity, soh_from_resistance)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DT = 1.0
Q_RATED = 50.0                                  # Ah
CELL_NOMINAL = BatteryModel(capacity_ah=Q_RATED)


def convergence_time(err: np.ndarray, tol: float = 0.05, dt: float = DT) -> float:
    """First time after which the error *stays* within ±tol."""
    within = np.abs(err) < tol
    for i in range(len(within)):
        if within[i:].all():
            return float(i * dt)
    return float("nan")


def largest_step_index(current: np.ndarray) -> int:
    """Index of the largest current step (best transient for R0 estimation)."""
    return int(np.argmax(np.abs(np.diff(current))))


# --------------------------------------------------------------------------- #
def run_soc_case():
    """SOC estimation with a mismatched model and a wrong initial SOC."""
    plant = BatteryModel(capacity_ah=Q_RATED)

    # EKF model: identified offline. Real identification errors are a few
    # percent on resistance and ~1-2 % on capacity; larger mismatches inject a
    # steady-state bias into SOC (R0 error x current / (dOCV/dSOC)).
    ekf_model = BatteryModel(capacity_ah=Q_RATED * 1.02,
                             r0=plant.r0 * 1.05,
                             r1=plant.r1 * 1.05, c1=plant.c1 * 0.95,
                             r2=plant.r2 * 1.05, c2=plant.c2 * 0.95)

    # Keep the SOC excursion inside the observable window (OCV is flat near
    # 100 % and near 0 %, where SOC is effectively unobservable).
    current = duty_cycle(duration_s=12000, dt=DT, base_load_a=8.0)
    truth = plant.simulate(current, dt=DT, soc0=0.95)

    v_meas = add_sensor_noise(truth["v"], sigma=0.005)      # 5 mV RMS
    i_meas = add_sensor_noise(current, sigma=0.05)          # 50 mA RMS

    # P0 reflects the initial uncertainty: we start 25 % off, so a 20 % std
    # (~0.04 variance) is a realistic prior.  A much larger P0 makes the filter
    # snap to the truth in a single step, which is not physical.
    ekf = SOCEstimatorEKF(ekf_model, dt=DT, soc0=0.75,
                          p0=[4e-2, 1e-4, 1e-4],
                          q=[1e-8, 1e-8, 1e-8], r=2.5e-5)
    soc_est = ekf.run(i_meas, v_meas)

    err = soc_est - truth["soc"]
    metrics = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_err": float(np.max(np.abs(err))),
        "final_err": float(abs(err[-1])),
        "conv_time_s": convergence_time(err),
        "soh_capacity": soh_from_capacity(
            estimate_capacity(i_meas, truth["soc"], DT), Q_RATED),
        "soh_resistance": soh_from_resistance(
            estimate_resistance_avg(v_meas, i_meas), CELL_NOMINAL.r0),
    }
    return current, truth, v_meas, soc_est, err, metrics


# --------------------------------------------------------------------------- #
def run_soh_case():
    """SOH estimation across a range of aged cells.

    The BMS still runs the *nominal* model (it does not know the cell aged),
    and estimates capacity from an OCV-calibrated SOC window — which is how
    the capacity method is applied in practice.
    """
    soh_true = np.array([1.00, 0.95, 0.90, 0.85, 0.80, 0.75])
    cap_est, res_est = [], []

    for s in soh_true:
        plant = aged_cell(BatteryModel(capacity_ah=Q_RATED), soh=float(s))
        current = duty_cycle(duration_s=12000, dt=DT)
        truth = plant.simulate(current, dt=DT, soc0=1.0)
        v_meas = add_sensor_noise(truth["v"], sigma=0.005)
        i_meas = add_sensor_noise(current, sigma=0.05)

        # Capacity from an OCV-calibrated SOC window (truth SOC = calibrated)
        cap_est.append(soh_from_capacity(
            estimate_capacity(i_meas, truth["soc"], DT), Q_RATED))

        # Resistance from the strongest transients (median of many steps)
        res_est.append(soh_from_resistance(
            estimate_resistance_avg(v_meas, i_meas), CELL_NOMINAL.r0))

    return soh_true, np.array(cap_est), np.array(res_est)


# --------------------------------------------------------------------------- #
def plot_soc(current, truth, v_meas, soc_est, err, metrics, t):
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(t, current, lw=0.6, color="#c0553a")
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("Duty cycle (positive = discharge)", fontsize=10)
    ax[0].grid(alpha=0.25)

    ax[1].plot(t, truth["soc"] * 100, lw=2.2, color="#2b4f9e", label="Ground truth")
    ax[1].plot(t, soc_est * 100, lw=1.6, ls="--", color="#e8a33d",
               label="EKF estimate")
    ax[1].set_ylabel("SOC [%]")
    ax[1].set_title(f"SOC tracking — RMSE {metrics['rmse']*100:.2f} %, "
                    f"max error {metrics['max_err']*100:.2f} % (mismatched model)",
                    fontsize=10)
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(alpha=0.25)

    ax[2].plot(t, err * 100, lw=1.0, color="#6b7280")
    ax[2].axhline(2, color="#c0553a", ls=":", lw=1)
    ax[2].axhline(-2, color="#c0553a", ls=":", lw=1)
    ax[2].set_ylabel("SOC error [%]")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_title("Estimation error (±2 % band)", fontsize=10)
    ax[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "soc_estimation.png", dpi=150)
    plt.close(fig)


def plot_soh(soh_true, cap_est, res_est):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(soh_true))
    w = 0.27

    ax.bar(x - w, soh_true * 100, w, label="Ground truth", color="#2b4f9e")
    ax.bar(x, cap_est * 100, w, label="Capacity-based", color="#3d8bfd")
    ax.bar(x + w, res_est * 100, w, label="Resistance-based", color="#e8a33d")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s*100:.0f} %" for s in soh_true])
    ax.set_xlabel("Cell ageing (ground-truth SOH)")
    ax.set_ylabel("Estimated SOH [%]")
    ax.set_title("SOH estimation: capacity vs resistance method", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "soh_estimation.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print("=" * 62)
    print("  SOC / SOH estimation — 2-RC model + Extended Kalman Filter")
    print("=" * 62)

    current, truth, v_meas, soc_est, err, metrics = run_soc_case()
    t = np.arange(len(current)) * DT
    plot_soc(current, truth, v_meas, soc_est, err, metrics, t)

    print("\n[SOC estimation] (EKF on a mismatched model, 25 % initial SOC error)")
    print(f"  RMSE        : {metrics['rmse']*100:.3f} %")
    print(f"  MAE         : {metrics['mae']*100:.3f} %")
    print(f"  Max error   : {metrics['max_err']*100:.3f} %")
    print(f"  Final error : {metrics['final_err']*100:.3f} %")
    print(f"  Convergence : {metrics['conv_time_s']:.0f} s")
    print(f"  SOH (capacity)  : {metrics['soh_capacity']*100:.1f} %")
    print(f"  SOH (resistance): {metrics['soh_resistance']*100:.1f} %")

    soh_true, cap_est, res_est = run_soh_case()
    plot_soh(soh_true, cap_est, res_est)

    print("\n[SOH estimation]")
    print("  ground truth | capacity-based | resistance-based")
    for s, c, r in zip(soh_true, cap_est, res_est):
        print(f"     {s*100:5.1f} %   |    {c*100:6.1f} %    |    {r*100:6.1f} %")

    summary = (
        "SOC estimation (2-RC ECM + EKF, mismatched model)\n"
        f"  RMSE      : {metrics['rmse']*100:.3f} %\n"
        f"  MAE       : {metrics['mae']*100:.3f} %\n"
        f"  Max error : {metrics['max_err']*100:.3f} %\n"
        f"  Convergence from 25 % initial error: {metrics['conv_time_s']:.0f} s\n"
        "\nSOH estimation\n"
        + "".join(f"  truth {s*100:5.1f} %  ->  capacity {c*100:6.1f} %  |  "
                  f"resistance {r*100:6.1f} %\n"
                  for s, c, r in zip(soh_true, cap_est, res_est))
    )
    (RESULTS / "metrics.txt").write_text(summary, encoding="utf-8")
    print(f"\nFigures + metrics written to {RESULTS}/")


if __name__ == "__main__":
    main()
