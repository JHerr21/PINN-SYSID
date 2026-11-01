# PINN-SYSID
Physics Informed System Identification for Model-Based Closed Loop Seizure Control

This project tests whether a better-identified model gives better closed-loop seizure control. A physics-informed neural network (PINN) and a joint unscented Kalman filter (UKF) each estimate the parameters of a Jansen-Rit neural mass model from simulated EEG. An LQR controller is then built on each identified model and judged on how well it suppresses seizure dynamics in the true model.

## Research questions

| ID | Question |
| --- | --- |
| H1 | Does the PINN recover parameters and hidden states better than the joint UKF? |
| H2 | Does the better-identified model give better suppression? |
| H3 | Can the PINN fit 256 Hz scalp EEG (CHB-MIT)? |

## Repository layout
 
```
PINN-SYSID/
  pyproject.toml
  configs/              # one YAML per experiment (nominal, sweep, control, mismatch, chbmit)
  src/sz/
    model.py            # Jansen-Rit vector field f(x, theta, p, u), sigmoid S, default params
    integrate.py        # RK4, Euler-Maruyama, decimation to 256 Hz
    data.py             # synthetic dataset generator, noise at set SNR, save/load
    identifiability.py  # error landscapes, sensitivity (jacfwd), Fisher information
    pinn.py             # windowed networks, losses, bounded parameters, training loop
    ukf.py              # sigma points, joint and fixed-parameter UKF, RTS smoother
    identify.py         # identify(data, cfg) -> theta_hat, x_hat, runtime
    control.py          # fixed point, linearization, discretization, LQR, saturation
    closed_loop.py      # plant + estimator + controller trial runner
    metrics.py          # parameter error, RMSE, line-length, band power, success, energy
    stats.py            # bootstrap, Wilson, Wilcoxon + Holm
    plots.py            # figures F1-F9
  experiments/
    01_simulator.py ... 07_chbmit.py
  results/              # one folder per run: config copy, git hash, seeds, metrics.json, arrays
  tests/                # model equations, integrator convergence, UKF vs filterpy, LQR stability
```
 
## Setup
 
All code is in PyTorch. PINN training is done on GPU. The simulator, UKF and controller run in float64 on the CPU.
