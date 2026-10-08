"""Diagnose the EKF under ideal conditions: constant current, no noise,
and an exactly known initial SOC.  Any error here is a code bug, not tuning."""
import sys
import numpy as np

sys.path.insert(0, "/Users/ma893399/projects/soc-soh-estimation/src")
from battery_model import BatteryModel          # noqa: E402
from ekf import SOCEstimatorEKF                 # noqa: E402

m = BatteryModel(capacity_ah=50.0)
dt = 1.0
I = np.full(1000, 10.0)                          # 10 A discharge for 1000 s

truth = m.simulate(I, dt=dt, soc0=1.0)

ekf = SOCEstimatorEKF(m, dt=dt, soc0=1.0, p0=[1e-6, 1e-6, 1e-6], r=2.5e-5)
est = ekf.run(I, truth["v"])

print("Ideal case: 10 A constant discharge, no noise, exact initial SOC")
print(f"  TRUE  final SOC : {truth['soc'][-1]:.4f}")
print(f"  EKF   final SOC : {est[-1]:.4f}")
print(f"  error           : {est[-1] - truth['soc'][-1]:+.5f}")
print(f"  max |error|     : {np.max(np.abs(est - truth['soc']))*100:.4f} %")
print()
print("  per-step check (first 10):")
print("    k    I[A]   V[V]     OCV(soc)   truth SOC   EKF SOC    err[%]")
for k in range(10):
    ocv_k = float(m.ocv(truth["soc"][k]))
    print(f"   {k:3d}  {I[k]:5.1f}  {truth['v'][k]:7.4f}  {ocv_k:8.4f}   "
          f"{truth['soc'][k]:8.4f}   {est[k]:8.4f}  {(est[k]-truth['soc'][k])*100:+7.3f}")
