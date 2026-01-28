# Continuous MaxSAT under Prequential Evaluation (GECCO 2026)

Reference paper: **“Continuous MaxSAT under Prequential Evaluation: A Population-Based Evolutionary Approach”**

This repo contains **Numba-accelerated** Python implementations of the paper’s prequential (test-then-train) streaming MaxSAT experiments, including:
- Synthetic streaming MaxSAT benchmark 
- Real-world clause streams (HWMCC and Linux Kconfig) 
- Time-budgeted comparison (equalized per-batch wall-clock)
- Replay memory study showing **catastrophic remembering** under adversarial drift 

---

#What the paper argues

In dynamic optimization, performance should be measured **online**: at each time step, evaluate on the incoming batch **before** adapting. This is **prequential evaluation** (test-then-train). :contentReference[oaicite:5]{index=5}

The paper shows:
- A simple **population-based genetic algorithm** (“PGA” / GA best-of-pop) provides **implicit memory via population diversity** and delivers stronger immediate prequential performance than single-solution local search baselines (WalkSAT / ProbSAT). 
- Adding explicit replay memory can *harm* immediate performance under adversarial drift, a failure mode called **catastrophic remembering**. 

---

## Repository layout

- `synthetic.py`  
  Main **synthetic streaming MaxSAT** experiment (concept drift + revisits). :contentReference[oaicite:8]{index=8}

- `synthetic_Memory_Replay.py`  
  **Replay-memory / catastrophic remembering** experiment (includes GA+Mem variant). :contentReference[oaicite:9]{index=9}

- `time_budget.py`  
  **Time-budgeted** per-batch comparison (hard cutoff: 10/20/30 ms). :contentReference[oaicite:10]{index=10}

- `real_ds.py`  
  **Real-world CNF stream** runner for directories of DIMACS `.cnf` files (HWMCC, Kconfig, or any CNF stream). :contentReference[oaicite:11]{index=11}

---

## Requirements

**Numba is REQUIRED** (there is no fallback). :contentReference[oaicite:12]{index=12}

Tested with Python 3.10+.

Install:
```bash
pip install numpy numba
