#!/usr/bin/env python3
"""
NUMBA-NIZED version of gecco_time_budget_per_batch_fixed2.py
Time-budgeted per-batch updates for streaming MaxSAT under prequential protocol.

- Numba is REQUIRED (no silent fallback).

- --numba_parallel : parallelize GA population scoring (prange)

"""

from __future__ import annotations
import argparse, math, random, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# -----------------------------
# REQUIRE Numba
# -----------------------------
try:
    from numba import njit, prange  # type: ignore
except Exception as e:
    raise SystemExit(
        "ERROR: Numba is required for this script but is not available.\n"
        "Install with: pip install numba\n"
        f"Original import error: {e}"
    )

Literal = int
Clause = Tuple[Literal, ...]  # kept for generator readability

# ---------------- Stream generator (same logic) ----------------
@dataclass
class StreamConfig:
    n_vars: int = 200
    k: int = 3
    batch_size: int = 200
    noise_frac: float = 0.15
    concepts: int = 3
    switch_period: int = 6
    revisit_pattern: Tuple[int, ...] = (0, 1, 2, 1)

def random_assignment_list(n: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(n)]

def build_concepts(cfg: StreamConfig) -> List[List[int]]:
    return [random_assignment_list(cfg.n_vars) for _ in range(cfg.concepts)]

def lit_value_list(lit: int, x: List[int]) -> bool:
    v = abs(lit) - 1
    val = (x[v] == 1)
    return val if lit > 0 else (not val)

def make_clause_satisfied_by(concept: List[int], n: int, k: int) -> Clause:
    vars_ = random.sample(range(1, n + 1), k)
    lits = [(v if random.random() < 0.5 else -v) for v in vars_]
    if not any(lit_value_list(l, concept) for l in lits):
        j = random.randrange(k)
        v = abs(lits[j])
        lits[j] = v if concept[v - 1] == 1 else -v
    return tuple(lits)

def make_random_clause(n: int, k: int) -> Clause:
    vars_ = random.sample(range(1, n + 1), k)
    return tuple(v if random.random() < 0.5 else -v for v in vars_)

def stream_batches(cfg: StreamConfig, steps: int, concepts: List[List[int]]):
    for t in range(steps):
        cid = cfg.revisit_pattern[(t // cfg.switch_period) % len(cfg.revisit_pattern)]
        concept = concepts[cid]
        m = cfg.batch_size
        m_noise = int(cfg.noise_frac * m)
        batch = [make_clause_satisfied_by(concept, cfg.n_vars, cfg.k) for _ in range(m - m_noise)]
        batch += [make_random_clause(cfg.n_vars, cfg.k) for _ in range(m_noise)]
        random.shuffle(batch)
        yield batch

def batch_to_array(batch: List[Clause], k: int) -> np.ndarray:
    arr = np.empty((len(batch), k), dtype=np.int32)
    for i, c in enumerate(batch):
        for j in range(k):
            arr[i, j] = int(c[j])
    return arr

# ---------------- SAT kernels (Numba) ----------------
@njit(cache=True)
def batch_score_u8(batch_lits: np.ndarray, x_u8: np.ndarray) -> int:
    m, k = batch_lits.shape
    sat = 0
    for i in range(m):
        ok = False
        for j in range(k):
            lit = batch_lits[i, j]
            v = abs(lit) - 1
            xv = x_u8[v]
            if (lit > 0 and xv == 1) or (lit < 0 and xv == 0):
                ok = True
                break
        if ok:
            sat += 1
    return sat

@njit(cache=True)
def pop_score_serial_u8(batch_lits: np.ndarray, pop_u8: np.ndarray) -> np.ndarray:
    out = np.empty(pop_u8.shape[0], dtype=np.int32)
    for i in range(pop_u8.shape[0]):
        out[i] = batch_score_u8(batch_lits, pop_u8[i])
    return out

@njit(cache=True, parallel=True)
def pop_score_parallel_u8(batch_lits: np.ndarray, pop_u8: np.ndarray) -> np.ndarray:
    out = np.empty(pop_u8.shape[0], dtype=np.int32)
    for i in prange(pop_u8.shape[0]):
        out[i] = batch_score_u8(batch_lits, pop_u8[i])
    return out

# ---------------- WalkSAT (Numba chunk) ----------------
@dataclass
class WalkSATConfig:
    p_random: float = 0.55

@njit(cache=True)
def walksat_chunk_u8(x_u8: np.ndarray, batch_lits: np.ndarray, flips: int, p_random: float, probe_tries: int) -> np.ndarray:
    """
    Do up to `flips` WalkSAT flips, returning the best assignment seen in this chunk.
    Uses "random unsat clause probing" then scan fallback to avoid building an unsat list.
    """
    m, k = batch_lits.shape
    best = x_u8.copy()
    best_sc = batch_score_u8(batch_lits, best)
    cur = x_u8.copy()

    for _ in range(flips):
        # find an unsatisfied clause index
        ci = -1
        for _t in range(probe_tries):
            r = np.random.randint(0, m)
            if batch_score_u8(batch_lits[r:r+1, :], cur) == 0:  # single-clause check via score==0
                ci = r
                break
        if ci < 0:
            for r in range(m):
                if batch_score_u8(batch_lits[r:r+1, :], cur) == 0:
                    ci = r
                    break
        if ci < 0:
            # all satisfied
            return cur

        clause = batch_lits[ci]

        if np.random.random() < p_random:
            lit = clause[np.random.randint(0, k)]
            v = abs(lit) - 1
        else:
            best_v = abs(clause[0]) - 1
            best_loc = -1
            for j in range(k):
                v2 = abs(clause[j]) - 1
                cur[v2] ^= 1
                sc = batch_score_u8(batch_lits, cur)
                cur[v2] ^= 1
                if sc > best_loc:
                    best_loc = sc
                    best_v = v2
            v = best_v

        cur[v] ^= 1
        sc = batch_score_u8(batch_lits, cur)
        if sc > best_sc:
            best_sc = sc
            best = cur.copy()
            if best_sc == m:
                return best

    return best

def walksat_update_time_budget_u8(x0_u8: np.ndarray, batch_lits: np.ndarray, cfg: WalkSATConfig,
                                 budget_s: float, chunk_flips: int, probe_tries: int) -> np.ndarray:
    deadline = time.perf_counter() + budget_s
    best = x0_u8.copy()
    best_sc = int(batch_score_u8(batch_lits, best))
    cur = x0_u8.copy()

    # Warm-up compile cost should be paid earlier; still fine.
    while time.perf_counter() < deadline:
        cand = walksat_chunk_u8(cur, batch_lits, chunk_flips, cfg.p_random, probe_tries)
        sc = int(batch_score_u8(batch_lits, cand))
        if sc > best_sc:
            best_sc = sc
            best = cand
            cur = cand
            if best_sc == batch_lits.shape[0]:
                break
        else:
            cur = cand
    return best

# ---------------- GA (Numba generation step) ----------------
@dataclass
class GAConfig:
    pop_size: int = 50
    elite_frac: float = 0.25
    mutation_rate: float = 0.01
    tournament_k: int = 2

@njit(cache=True)
def tournament_pick(fits: np.ndarray, k: int) -> int:
    k = k if k <= fits.shape[0] else fits.shape[0]
    best = np.random.randint(0, fits.shape[0])
    best_fit = fits[best]
    for _ in range(k - 1):
        cand = np.random.randint(0, fits.shape[0])
        f = fits[cand]
        if f > best_fit:
            best = cand
            best_fit = f
    return best

@njit(cache=True)
def ga_next_generation_rng(pop: np.ndarray, fits: np.ndarray, elite_n: int, tournament_k: int,
                           mutation_rate: float) -> np.ndarray:
    """
    One GA generation:
    - Elitism
    - Tournament selection
    - One-point crossover
    - Bit-flip mutation
    Ensures output population size equals input size.
    """
    P, V = pop.shape
    order = np.argsort(fits)[::-1]
    new_pop = np.empty_like(pop)

    # elites
    for i in range(elite_n):
        src = order[i]
        for v in range(V):
            new_pop[i, v] = pop[src, v]

    # fill rest
    for row in range(elite_n, P):
        p1 = tournament_pick(fits, tournament_k)
        p2 = tournament_pick(fits, tournament_k)
        cut = np.random.randint(1, V)

        for v in range(cut):
            new_pop[row, v] = pop[p1, v]
        for v in range(cut, V):
            new_pop[row, v] = pop[p2, v]

        for v in range(V):
            if np.random.random() < mutation_rate:
                new_pop[row, v] ^= 1

    return new_pop

def ga_step_time_budget_u8(pop_u8: np.ndarray, batch_lits: np.ndarray, cfg: GAConfig,
                           budget_s: float, numba_parallel: bool) -> np.ndarray:
    """
    Run as many generations as fit in the time budget, maintaining pop size invariant.
    """
    deadline = time.perf_counter() + budget_s
    P = pop_u8.shape[0]
    elite_n = max(1, int(round(cfg.elite_frac * cfg.pop_size)))
    elite_n = min(elite_n, P)

    while time.perf_counter() < deadline:
        fits = pop_score_parallel_u8(batch_lits, pop_u8) if numba_parallel else pop_score_serial_u8(batch_lits, pop_u8)
        pop_u8 = ga_next_generation_rng(pop_u8, fits.astype(np.int32), elite_n, cfg.tournament_k, cfg.mutation_rate)

    return pop_u8

# ---------------- Stats (same as original) ----------------
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def wilcoxon(x: List[float], y: List[float]) -> Dict[str, float]:
    diffs = [a - b for a, b in zip(x, y) if (a - b) != 0.0]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "W": 0.0, "z": 0.0, "p": 1.0}

    ranked = sorted((abs(d), d) for d in diffs)
    Wp = sum(i + 1 for i, (_, d) in enumerate(ranked) if d > 0)
    Wm = sum(i + 1 for i, (_, d) in enumerate(ranked) if d < 0)
    W = min(Wp, Wm)

    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu + 0.5) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": float(n), "W": float(W), "z": float(z), "p": float(p)}

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--budget_ms", type=float, default=10.0)
    ap.add_argument("--numba_parallel", action="store_true", help="parallelize GA population scoring (prange)")
    ap.add_argument("--ws_chunk_flips", type=int, default=128, help="WalkSAT flips per budget chunk")
    ap.add_argument("--ws_probe_tries", type=int, default=32, help="random unsat-clause probes before scan")
    args = ap.parse_args()

    stream_cfg = StreamConfig()
    ws_cfg = WalkSATConfig()
    ga_cfg = GAConfig()

    ws_scores: List[float] = []
    ga_scores: List[float] = []

    print(f"Protocol: {args.runs} runs, time budget {args.budget_ms} ms per batch")
    if args.numba_parallel:
        print("Numba parallel: ON (population scoring uses prange)")

    # Warmup JIT outside timing (compile once)
    np.random.seed(0)
    dummy_batch = np.array([[1, -2, 3]], dtype=np.int32)
    dummy_x = (np.random.rand(stream_cfg.n_vars) < 0.5).astype(np.uint8)
    _ = batch_score_u8(dummy_batch, dummy_x)
    _ = walksat_chunk_u8(dummy_x, dummy_batch, 1, ws_cfg.p_random, 1)
    dummy_pop = (np.random.rand(ga_cfg.pop_size, stream_cfg.n_vars) < 0.5).astype(np.uint8)
    _ = pop_score_serial_u8(dummy_batch, dummy_pop[:min(ga_cfg.pop_size, 4)])
    _ = ga_next_generation_rng(dummy_pop[:min(ga_cfg.pop_size, 8)], np.ones(min(ga_cfg.pop_size, 8), dtype=np.int32),
                               1, ga_cfg.tournament_k, ga_cfg.mutation_rate)

    for r in range(args.runs):
        seed = r + 1
        random.seed(seed)
        np.random.seed(seed)

        concepts = build_concepts(stream_cfg)

        ws_u8 = (np.random.rand(stream_cfg.n_vars) < 0.5).astype(np.uint8)
        ga_pop = (np.random.rand(ga_cfg.pop_size, stream_cfg.n_vars) < 0.5).astype(np.uint8)

        s_ws = 0.0
        s_ga = 0.0

        budget_s = args.budget_ms / 1000.0

        for batch in stream_batches(stream_cfg, args.steps, concepts):
            batch_lits = batch_to_array(batch, stream_cfg.k)

            # PREQUENTIAL TEST
            s_ws += batch_score_u8(batch_lits, ws_u8) / stream_cfg.batch_size

            fits = pop_score_parallel_u8(batch_lits, ga_pop) if args.numba_parallel else pop_score_serial_u8(batch_lits, ga_pop)
            best = int(fits.max()) if fits.size else 0
            s_ga += best / stream_cfg.batch_size

            # UPDATE (same time budget)
            ws_u8 = walksat_update_time_budget_u8(ws_u8, batch_lits, ws_cfg, budget_s, args.ws_chunk_flips, args.ws_probe_tries)
            ga_pop = ga_step_time_budget_u8(ga_pop, batch_lits, ga_cfg, budget_s, args.numba_parallel)

        ws_scores.append(s_ws / args.steps)
        ga_scores.append(s_ga / args.steps)
        print(f"RUN {r:02d}  seed={seed}  WS={ws_scores[-1]:.4f}  GA={ga_scores[-1]:.4f}")

    res = wilcoxon(ga_scores, ws_scores)
    print("\nSummary:")
    print(f"WS mean={mean(ws_scores):.4f}  std={stdev(ws_scores):.4f}")
    print(f"GA mean={mean(ga_scores):.4f}  std={stdev(ga_scores):.4f}")
    print(f"Wilcoxon (GA vs WS): n={int(res['n'])}  W={res['W']:.2f}  z={res['z']:.3f}  p≈{res['p']:.4g}")

if __name__ == "__main__":
    main()
