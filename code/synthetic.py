#!/usr/bin/env python3
"""

Numba is REQUIRED (no fallback).

Tip:
- For best speed, use --numba_parallel (parallel population scoring).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------
# REQUIRE Numba
# ----------------------------
try:
    from numba import njit, prange  # type: ignore
except Exception as e:
    raise SystemExit(
        "ERROR: Numba is required for this script but is not available.\n"
        "Install with: pip install numba\n"
        f"Original import error: {e}"
    )

Literal = int
Clause = Tuple[Literal, ...]  # stream gen readability

# ----------------------------
# Stream generator (same spirit)
# ----------------------------
@dataclass
class StreamConfig:
    n_vars: int = 200
    k: int = 3
    batch_size: int = 200
    noise_frac: float = 0.15
    concepts: int = 3
    switch_period: int = 6
    revisit_pattern: Tuple[int, ...] = (0, 1, 2, 1)

def random_assignment_u8(n: int) -> np.ndarray:
    return (np.random.rand(n) < 0.5).astype(np.uint8)

def build_concepts(cfg: StreamConfig) -> List[np.ndarray]:
    return [random_assignment_u8(cfg.n_vars) for _ in range(cfg.concepts)]

def lit_value_u8(lit: int, concept_u8: np.ndarray) -> bool:
    v = abs(lit) - 1
    val = (concept_u8[v] == 1)
    return val if lit > 0 else (not val)

def make_clause_satisfied_by(concept: np.ndarray, n: int, k: int) -> Clause:
    vars_ = random.sample(range(1, n + 1), k)
    lits: List[int] = []
    for v in vars_:
        sign = 1 if random.random() < 0.5 else -1
        lits.append(sign * v)
    if not any(lit_value_u8(l, concept) for l in lits):
        j = random.randrange(k)
        v = abs(lits[j])
        lits[j] = (1 if concept[v - 1] == 1 else -1) * v
    return tuple(lits)

def make_random_clause(n: int, k: int) -> Clause:
    vars_ = random.sample(range(1, n + 1), k)
    return tuple((1 if random.random() < 0.5 else -1) * v for v in vars_)

def stream_batches(cfg: StreamConfig, steps: int, concepts: List[np.ndarray]):
    for t in range(steps):
        cid = cfg.revisit_pattern[(t // cfg.switch_period) % len(cfg.revisit_pattern)]
        concept = concepts[cid]
        m = cfg.batch_size
        m_noise = int(round(cfg.noise_frac * m))
        m_sig = m - m_noise
        batch = [make_clause_satisfied_by(concept, cfg.n_vars, cfg.k) for _ in range(m_sig)]
        batch += [make_random_clause(cfg.n_vars, cfg.k) for _ in range(m_noise)]
        random.shuffle(batch)
        yield t, cid, batch

def batch_to_array(batch: List[Clause], k: int) -> np.ndarray:
    arr = np.empty((len(batch), k), dtype=np.int32)
    for i, c in enumerate(batch):
        for j in range(k):
            arr[i, j] = int(c[j])
    return arr

# ----------------------------
# Numba scoring kernels (uint8 assignments)
# ----------------------------
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

@njit(cache=True)
def clause_satisfied_u8(clause_lits: np.ndarray, x_u8: np.ndarray) -> bool:
    for j in range(clause_lits.shape[0]):
        lit = clause_lits[j]
        v = abs(lit) - 1
        xv = x_u8[v]
        if (lit > 0 and xv == 1) or (lit < 0 and xv == 0):
            return True
    return False

# ----------------------------
# WalkSAT (NOW NUMBA)
# ----------------------------
@dataclass
class WalkSATConfig:
    flips: int = 5000
    p_random: float = 0.55
    restarts: int = 1
    # how many random clause attempts before giving up and scanning for an unsat clause
    unsat_probe_tries: int = 32


### PROBSAT ADDITION
@dataclass
class ProbSATConfig:
    flips: int = 5000
    noise: float = 0.57
    restarts: int = 1
    unsat_probe_tries: int = 32

@njit(cache=True)
def random_assignment_u8_numba(n: int) -> np.ndarray:
    x = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x[i] = 1 if np.random.random() < 0.5 else 0
    return x

@njit(cache=True)
def find_unsat_clause_index(batch_lits: np.ndarray, x_u8: np.ndarray, probe_tries: int) -> int:
    m = batch_lits.shape[0]
    # probe a few random clauses first
    for _ in range(probe_tries):
        ci = np.random.randint(0, m)
        if not clause_satisfied_u8(batch_lits[ci], x_u8):
            return ci
    # fallback scan
    for ci in range(m):
        if not clause_satisfied_u8(batch_lits[ci], x_u8):
            return ci
    return -1  # all satisfied

@njit(cache=True)
def walksat_one_restart_numba(
    start_x: np.ndarray,
    batch_lits: np.ndarray,
    flips: int,
    p_random: float,
    unsat_probe_tries: int,
) -> np.ndarray:
    x = start_x.copy()
    m, k = batch_lits.shape
    for _ in range(flips):
        ci = find_unsat_clause_index(batch_lits, x, unsat_probe_tries)
        if ci < 0:
            break  # solved (for this batch)
        clause = batch_lits[ci]

        if np.random.random() < p_random:
            lit = clause[np.random.randint(0, k)]
            v = abs(lit) - 1
        else:
            # greedy among literals in the chosen unsat clause
            best_v = abs(clause[0]) - 1
            best_sc = -1
            for j in range(k):
                v2 = abs(clause[j]) - 1
                x[v2] ^= 1
                sc = batch_score_u8(batch_lits, x)
                x[v2] ^= 1
                if sc > best_sc:
                    best_sc = sc
                    best_v = v2
            v = best_v

        x[v] ^= 1
    return x

### PROBSAT ADDITION
@njit(cache=True)
def probsat_break_count_var(batch_lits: np.ndarray, x: np.ndarray, v: int) -> int:
    # Count how many currently SAT clauses would become UNSAT if variable v is flipped.
    # This is the canonical 'break' measure used by ProbSAT-style selection rules.
    m, k = batch_lits.shape
    brk = 0
    for i in range(m):
        sat_count = 0
        sat_by_v = 0
        for j in range(k):
            lit = batch_lits[i, j]
            var = abs(lit) - 1
            xv = x[var]
            is_sat = (lit > 0 and xv == 1) or (lit < 0 and xv == 0)
            if is_sat:
                sat_count += 1
                if var == v:
                    sat_by_v += 1
        # Clause breaks iff it is currently satisfied exclusively by literals of v
        if sat_by_v > 0 and sat_count == sat_by_v:
            brk += 1
    return brk

@njit(cache=True)
def probsat_pick_variable(clause: np.ndarray, batch_lits: np.ndarray, x: np.ndarray, cb: float) -> int:
    # Canonical ProbSAT variable choice from an UNSAT clause.
    # Probability for candidate var v_i is proportional to (break(v_i) + 1)^(-cb).
    # Here cb is the 'exponent' parameter (higher -> more greedy toward low-break vars).
    k = clause.shape[0]
    weights = np.empty(k, dtype=np.float64)

    for i in range(k):
        v = abs(clause[i]) - 1
        brk = probsat_break_count_var(batch_lits, x, v)
        weights[i] = 1.0 / ((brk + 1.0) ** cb)

    s = weights.sum()
    if s <= 0.0:
        return abs(clause[np.random.randint(0, k)]) - 1

    r = np.random.random() * s
    acc = 0.0
    for i in range(k):
        acc += weights[i]
        if acc >= r:
            return abs(clause[i]) - 1

    return abs(clause[0]) - 1

@njit(cache=True)
def probsat_one_restart_numba(
    start_x: np.ndarray,
    batch_lits: np.ndarray,
    flips: int,
    noise: float,
    unsat_probe_tries: int,
) -> np.ndarray:
    x = start_x.copy()
    for _ in range(flips):
        ci = find_unsat_clause_index(batch_lits, x, unsat_probe_tries)
        if ci < 0:
            break
        clause = batch_lits[ci]
        v = probsat_pick_variable(clause, batch_lits, x, noise)
        x[v] ^= 1
    return x

def walksat_update_u8(x: np.ndarray, batch_lits: np.ndarray, n: int, cfg: WalkSATConfig) -> np.ndarray:
    # keep restart logic in Python (small overhead) but each restart is Numba-fast
    best = x.copy()
    best_score = int(batch_score_u8(batch_lits, best))

    for r in range(cfg.restarts):
        start = x if r == 0 else random_assignment_u8(n)
        cand = walksat_one_restart_numba(start, batch_lits, cfg.flips, cfg.p_random, cfg.unsat_probe_tries)
        sc = int(batch_score_u8(batch_lits, cand))
        if sc > best_score:
            best_score = sc
            best = cand
            if best_score == batch_lits.shape[0]:
                break
    return best

### PROBSAT ADDITION
def probsat_update_u8(x: np.ndarray, batch_lits: np.ndarray, n: int, cfg: ProbSATConfig) -> np.ndarray:
    # restart logic in Python (small overhead) but each restart is Numba-fast
    best = x.copy()
    best_score = int(batch_score_u8(batch_lits, best))

    for r in range(cfg.restarts):
        start = x if r == 0 else random_assignment_u8(n)
        cand = probsat_one_restart_numba(start, batch_lits, cfg.flips, cfg.noise, cfg.unsat_probe_tries)
        sc = int(batch_score_u8(batch_lits, cand))
        if sc > best_score:
            best_score = sc
            best = cand
            if best_score == batch_lits.shape[0]:
                break
    return best

# ----------------------------
# GA (Numba next-generation with rng)
# ----------------------------
@dataclass
class GAConfig:
    pop_size: int = 80
    elite_frac: float = 0.15
    mutation_rate: float = 0.01
    tournament_k: int = 3
    generations_per_step: int = 4
    crossover_rate: float = 0.9

@njit(cache=True)
def tournament_pick(fits: np.ndarray, k: int) -> int:
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
                           crossover_rate: float, mutation_rate: float) -> np.ndarray:
    P, V = pop.shape
    order = np.argsort(fits)[::-1]
    new_pop = np.empty_like(pop)

    for i in range(elite_n):
        src = order[i]
        for v in range(V):
            new_pop[i, v] = pop[src, v]

    for row in range(elite_n, P):
        p1 = tournament_pick(fits, tournament_k)
        p2 = tournament_pick(fits, tournament_k)

        if np.random.random() < crossover_rate:
            cut = np.random.randint(1, V)
            for v in range(cut):
                new_pop[row, v] = pop[p1, v]
            for v in range(cut, V):
                new_pop[row, v] = pop[p2, v]
        else:
            for v in range(V):
                new_pop[row, v] = pop[p1, v]

        for v in range(V):
            if np.random.random() < mutation_rate:
                new_pop[row, v] ^= 1

    return new_pop

def ga_step_u8(pop: np.ndarray, batch_lits: np.ndarray, cfg: GAConfig, numba_parallel: bool) -> np.ndarray:
    elite_n = max(1, int(round(cfg.elite_frac * cfg.pop_size)))
    for _ in range(cfg.generations_per_step):
        fits = pop_score_parallel_u8(batch_lits, pop) if numba_parallel else pop_score_serial_u8(batch_lits, pop)
        pop = ga_next_generation_rng(pop, fits.astype(np.int32), elite_n, cfg.tournament_k, cfg.crossover_rate, cfg.mutation_rate)
    return pop

# ----------------------------
# Stats helpers (unchanged)
# ----------------------------
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def ci95(xs: List[float]) -> Tuple[float, float]:
    m = mean(xs)
    s = stdev(xs)
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    if s == 0.0:
        return (m, m)
    half = 1.96 * (s / math.sqrt(n))
    return (m - half, m + half)

def wilcoxon_signed_rank_pvalue(x: List[float], y: List[float]) -> Dict[str, float]:
    if len(x) != len(y):
        raise ValueError("wilcoxon requires paired lists of equal length")
    diffs = [(xi - yi) for xi, yi in zip(x, y)]
    nz = [(abs(d), d) for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return {"n": 0, "W": 0.0, "z": 0.0, "p": 1.0}
    nz.sort(key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and nz[j][0] == nz[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    W_plus = 0.0
    W_minus = 0.0
    for r, (_, d) in zip(ranks, nz):
        if d > 0:
            W_plus += r
        else:
            W_minus += r
    W = min(W_plus, W_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0.0:
        return {"n": n, "W": W, "z": 0.0, "p": 1.0}
    z = (W - mu + 0.5) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": n, "W": W, "z": z, "p": p}

# ----------------------------
# One run (one seed)
# ----------------------------
@dataclass
class RunResult:
    mean_ws_warm: float
    mean_ws_restart: float
    mean_ws_eps: float
    mean_ps_warm: float
    mean_ps_restart: float
    mean_ps_eps: float
    mean_ga: float
    t_ws_warm: float
    t_ws_restart: float
    t_ws_eps: float
    t_ps_warm: float
    t_ps_restart: float
    t_ps_eps: float
    t_ga: float

def run_one(seed: int, steps: int, epsilon: float,
            stream_cfg: StreamConfig, ws_cfg: WalkSATConfig, ps_cfg: ProbSATConfig, ga_cfg: GAConfig,
            numba_parallel: bool) -> RunResult:
    random.seed(seed)
    np.random.seed(seed)

    concepts = build_concepts(stream_cfg)

    ws_warm = random_assignment_u8(stream_cfg.n_vars)
    ws_eps = random_assignment_u8(stream_cfg.n_vars)
    ps_warm = random_assignment_u8(stream_cfg.n_vars)
    ps_eps = random_assignment_u8(stream_cfg.n_vars)
    ga_pop = (np.random.rand(ga_cfg.pop_size, stream_cfg.n_vars) < 0.5).astype(np.uint8)

    s_ws_warm = s_ws_restart = s_ws_eps = 0.0
    s_ps_warm = s_ps_restart = s_ps_eps = 0.0
    s_ga = 0.0
    t_ws_warm = t_ws_restart = t_ws_eps = 0.0
    t_ps_warm = t_ps_restart = t_ps_eps = 0.0
    t_ga = 0.0
    B = float(stream_cfg.batch_size)

    # JIT warmup
    dummy = np.array([[1, -2, 3]], dtype=np.int32)
    _ = batch_score_u8(dummy, ws_warm)
    _ = walksat_one_restart_numba(ws_warm, dummy, 1, 0.5, 1)
    _ = probsat_one_restart_numba(ws_warm, dummy, 1, 0.57, 1)
    _ = ga_next_generation_rng(ga_pop[:min(ga_cfg.pop_size, 8)], np.ones(min(ga_cfg.pop_size, 8), dtype=np.int32),
                               1, ga_cfg.tournament_k, ga_cfg.crossover_rate, ga_cfg.mutation_rate)

    for _, _, batch in stream_batches(stream_cfg, steps, concepts):
        batch_lits = batch_to_array(batch, stream_cfg.k)

        # PREQUENTIAL TEST
        s_ws_warm += batch_score_u8(batch_lits, ws_warm) / B

        ws_restart_x0 = random_assignment_u8(stream_cfg.n_vars)
        s_ws_restart += batch_score_u8(batch_lits, ws_restart_x0) / B

        s_ws_eps += batch_score_u8(batch_lits, ws_eps) / B

        # ProbSAT PREQUENTIAL TEST
        s_ps_warm += batch_score_u8(batch_lits, ps_warm) / B

        ps_restart_x0 = random_assignment_u8(stream_cfg.n_vars)
        s_ps_restart += batch_score_u8(batch_lits, ps_restart_x0) / B

        s_ps_eps += batch_score_u8(batch_lits, ps_eps) / B

        fits = pop_score_parallel_u8(batch_lits, ga_pop) if numba_parallel else pop_score_serial_u8(batch_lits, ga_pop)
        s_ga += float(int(fits.max()) if fits.size else 0) / B

        # UPDATE (timed)
        t0 = time.perf_counter()
        ws_warm = walksat_update_u8(ws_warm, batch_lits, stream_cfg.n_vars, ws_cfg)
        t_ws_warm += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = walksat_update_u8(ws_restart_x0, batch_lits, stream_cfg.n_vars, ws_cfg)
        t_ws_restart += time.perf_counter() - t0

        if random.random() < epsilon:
            ws_eps = random_assignment_u8(stream_cfg.n_vars)
        t0 = time.perf_counter()
        ws_eps = walksat_update_u8(ws_eps, batch_lits, stream_cfg.n_vars, ws_cfg)
        t_ws_eps += time.perf_counter() - t0

        # ProbSAT UPDATE (timed)
        t0 = time.perf_counter()
        ps_warm = probsat_update_u8(ps_warm, batch_lits, stream_cfg.n_vars, ps_cfg)
        t_ps_warm += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = probsat_update_u8(ps_restart_x0, batch_lits, stream_cfg.n_vars, ps_cfg)
        t_ps_restart += time.perf_counter() - t0

        if random.random() < epsilon:
            ps_eps = random_assignment_u8(stream_cfg.n_vars)
        t0 = time.perf_counter()
        ps_eps = probsat_update_u8(ps_eps, batch_lits, stream_cfg.n_vars, ps_cfg)
        t_ps_eps += time.perf_counter() - t0

        t0 = time.perf_counter()
        ga_pop = ga_step_u8(ga_pop, batch_lits, ga_cfg, numba_parallel)
        t_ga += time.perf_counter() - t0

    return RunResult(
        mean_ws_warm=s_ws_warm / steps,
        mean_ws_restart=s_ws_restart / steps,
        mean_ws_eps=s_ws_eps / steps,
        mean_ps_warm=s_ps_warm / steps,
        mean_ps_restart=s_ps_restart / steps,
        mean_ps_eps=s_ps_eps / steps,
        mean_ga=s_ga / steps,
        t_ws_warm=t_ws_warm,
        t_ws_restart=t_ws_restart,
        t_ws_eps=t_ws_eps,
        t_ps_warm=t_ps_warm,
        t_ps_restart=t_ps_restart,
        t_ps_eps=t_ps_eps,
        t_ga=t_ga,
    )

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--numba_parallel", action="store_true")

    # Stream parameters
    ap.add_argument("--n_vars", type=int, default=200, help="number of Boolean variables")
    ap.add_argument("--k", type=int, default=3, help="literals per clause")
    ap.add_argument("--batch_size", type=int, default=200, help="clauses per batch")
    ap.add_argument("--noise_frac", type=float, default=0.15, help="fraction of random (noise) clauses per batch")
    ap.add_argument("--concepts", type=int, default=3, help="number of underlying concepts")
    ap.add_argument("--switch_period", type=int, default=6, help="batches per concept segment before indexing revisit_pattern")
    ap.add_argument(
        "--revisit_pattern",
        type=str,
        default="0,1,2,1",
        help="comma-separated concept id pattern, e.g. '0,1,2,1'",
    )

    # WalkSAT parameters
    ap.add_argument("--ws_flips", type=int, default=5000, help="WalkSAT flips per batch")
    ap.add_argument("--ws_p_random", type=float, default=0.55, help="WalkSAT random-move probability")
    ap.add_argument("--ws_restarts", type=int, default=1, help="WalkSAT restarts per batch")
    ap.add_argument(
        "--ws_unsat_probe_tries",
        type=int,
        default=32,
        help="WalkSAT: random unsat-clause probes before full scan (Numba kernel)",
    )

    # ProbSAT parameters
    ap.add_argument("--ps_flips", type=int, default=5000, help="ProbSAT flips per batch")
    ap.add_argument("--ps_noise", type=float, default=2.06, help="ProbSAT cb exponent (higher = greedier toward low-break variables)")
    ap.add_argument("--ps_restarts", type=int, default=1, help="ProbSAT restarts per batch")
    ap.add_argument("--ps_unsat_probe_tries", type=int, default=32, help="ProbSAT: random unsat-clause probes before full scan (Numba kernel)")


    # GA parameters
    ap.add_argument("--ga_pop_size", type=int, default=80, help="GA population size")
    ap.add_argument("--ga_elite_frac", type=float, default=0.15, help="GA elite fraction")
    ap.add_argument("--ga_mutation_rate", type=float, default=0.01, help="GA mutation rate per bit")
    ap.add_argument("--ga_tournament_k", type=int, default=3, help="GA tournament size")
    ap.add_argument("--ga_generations_per_step", type=int, default=4, help="GA generations per batch")
    ap.add_argument("--ga_crossover_rate", type=float, default=0.9, help="GA crossover probability")

    args = ap.parse_args()

    # Basic validation
    if args.n_vars <= 0:
        raise SystemExit("ERROR: --n_vars must be > 0")
    if args.k <= 0:
        raise SystemExit("ERROR: --k must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("ERROR: --batch_size must be > 0")
    if not (0.0 <= args.noise_frac <= 1.0):
        raise SystemExit("ERROR: --noise_frac must be in [0, 1]")
    if args.concepts <= 0:
        raise SystemExit("ERROR: --concepts must be > 0")
    if args.switch_period <= 0:
        raise SystemExit("ERROR: --switch_period must be > 0")
    if args.ws_flips <= 0:
        raise SystemExit("ERROR: --ws_flips must be > 0")
    if not (0.0 <= args.ws_p_random <= 1.0):
        raise SystemExit("ERROR: --ws_p_random must be in [0, 1]")
    if args.ws_restarts <= 0:
        raise SystemExit("ERROR: --ws_restarts must be > 0")
    if args.ws_unsat_probe_tries <= 0:
        raise SystemExit("ERROR: --ws_unsat_probe_tries must be > 0")
    if args.ps_flips <= 0:
        raise SystemExit("ERROR: --ps_flips must be > 0")
    if args.ps_noise <= 0.0:
        raise SystemExit("ERROR: --ps_noise must be > 0")
    if args.ps_restarts <= 0:
        raise SystemExit("ERROR: --ps_restarts must be > 0")
    if args.ps_unsat_probe_tries <= 0:
        raise SystemExit("ERROR: --ps_unsat_probe_tries must be > 0")
    if args.ga_pop_size <= 0:
        raise SystemExit("ERROR: --ga_pop_size must be > 0")
    if not (0.0 < args.ga_elite_frac <= 1.0):
        raise SystemExit("ERROR: --ga_elite_frac must be in (0, 1]")
    if not (0.0 <= args.ga_mutation_rate <= 1.0):
        raise SystemExit("ERROR: --ga_mutation_rate must be in [0, 1]")
    if args.ga_tournament_k <= 0:
        raise SystemExit("ERROR: --ga_tournament_k must be > 0")
    if args.ga_generations_per_step <= 0:
        raise SystemExit("ERROR: --ga_generations_per_step must be > 0")
    if not (0.0 <= args.ga_crossover_rate <= 1.0):
        raise SystemExit("ERROR: --ga_crossover_rate must be in [0, 1]")

    # Parse revisit pattern
    try:
        revisit_pattern = tuple(int(s.strip()) for s in args.revisit_pattern.split(",") if s.strip() != "")
    except Exception:
        raise SystemExit("ERROR: --revisit_pattern must be comma-separated integers, e.g. '0,1,2,1'")
    if len(revisit_pattern) == 0:
        raise SystemExit("ERROR: --revisit_pattern must contain at least one integer")
    if any((cid < 0 or cid >= args.concepts) for cid in revisit_pattern):
        raise SystemExit("ERROR: --revisit_pattern contains concept id outside [0, concepts-1]")

    stream_cfg = StreamConfig(
        n_vars=args.n_vars,
        k=args.k,
        batch_size=args.batch_size,
        noise_frac=args.noise_frac,
        concepts=args.concepts,
        switch_period=args.switch_period,
        revisit_pattern=revisit_pattern,
    )
    ws_cfg = WalkSATConfig(
        flips=args.ws_flips,
        p_random=args.ws_p_random,
        restarts=args.ws_restarts,
        unsat_probe_tries=args.ws_unsat_probe_tries,
    )
    ps_cfg = ProbSATConfig(
        flips=args.ps_flips,
        noise=args.ps_noise,
        restarts=args.ps_restarts,
        unsat_probe_tries=args.ps_unsat_probe_tries,
    )
    ga_cfg = GAConfig(
        pop_size=args.ga_pop_size,
        elite_frac=args.ga_elite_frac,
        mutation_rate=args.ga_mutation_rate,
        tournament_k=args.ga_tournament_k,
        generations_per_step=args.ga_generations_per_step,
        crossover_rate=args.ga_crossover_rate,
    )

    ws_warm_means: List[float] = []
    ws_restart_means: List[float] = []
    ws_eps_means: List[float] = []
    ga_means: List[float] = []

    ps_warm_means: List[float] = []
    ps_restart_means: List[float] = []
    ps_eps_means: List[float] = []

    ws_warm_times: List[float] = []
    ws_restart_times: List[float] = []
    ws_eps_times: List[float] = []
    ga_times: List[float] = []

    ps_warm_times: List[float] = []
    ps_restart_times: List[float] = []
    ps_eps_times: List[float] = []

    if args.numba_parallel:
        print("Numba parallel: ON (population scoring uses prange)")

    for i in range(args.runs):
        seed = args.seed0 + i
        rr = run_one(seed, args.steps, args.epsilon, stream_cfg, ws_cfg, ps_cfg, ga_cfg, args.numba_parallel)
        print(
            f"RUN {i:02d}  seed={seed}  "
            f"WS_warm={rr.mean_ws_warm:.4f}  "
            f"WS_restart={rr.mean_ws_restart:.4f}  "
            f"WS_eps={rr.mean_ws_eps:.4f}  "
            f"PS_warm={rr.mean_ps_warm:.4f}  "
            f"PS_restart={rr.mean_ps_restart:.4f}  "
            f"PS_eps={rr.mean_ps_eps:.4f}  "
            f"GA={rr.mean_ga:.4f}  "
            f"t_WS={rr.t_ws_warm:.2f}s  "
            f"t_GA={rr.t_ga:.2f}s"
        )

        ws_warm_means.append(rr.mean_ws_warm)
        ws_restart_means.append(rr.mean_ws_restart)
        ws_eps_means.append(rr.mean_ws_eps)
        ga_means.append(rr.mean_ga)

        ps_warm_means.append(rr.mean_ps_warm)
        ps_restart_means.append(rr.mean_ps_restart)
        ps_eps_means.append(rr.mean_ps_eps)

        ws_warm_times.append(rr.t_ws_warm)
        ws_restart_times.append(rr.t_ws_restart)
        ws_eps_times.append(rr.t_ws_eps)
        ga_times.append(rr.t_ga)

        ps_warm_times.append(rr.t_ps_warm)
        ps_restart_times.append(rr.t_ps_restart)
        ps_eps_times.append(rr.t_ps_eps)

    print(f"Protocol: All results are averaged over {args.runs} independent runs with different random seeds.")
    print(f"Per run: steps={args.steps}, batch_size={stream_cfg.batch_size}, noise_frac={stream_cfg.noise_frac}, epsilon={args.epsilon}\n")

    def fmt_stats(name: str, xs: List[float]) -> str:
        m = mean(xs); s = stdev(xs); lo, hi = ci95(xs)
        return f"{name:<28} mean={m:.3f}  std={s:.3f}  95%CI=[{lo:.3f},{hi:.3f}]"

    print("Prequential satisfaction (per-run mean over batches):")
    print(fmt_stats("WalkSAT warm-start", ws_warm_means))
    print(fmt_stats("WalkSAT restart/batch", ws_restart_means))
    print(fmt_stats("WalkSAT epsilon-restart", ws_eps_means))
    print(fmt_stats("ProbSAT warm-start", ps_warm_means))
    print(fmt_stats("ProbSAT restart/batch", ps_restart_means))
    print(fmt_stats("ProbSAT epsilon-restart", ps_eps_means))
    print(fmt_stats("GA (best-of-pop)", ga_means))

    def fmt_time(name: str, xs: List[float]) -> str:
        m = mean(xs); s = stdev(xs); lo, hi = ci95(xs)
        return f"{name:<28} mean={m:.2f}s std={s:.2f}s  95%CI=[{lo:.2f},{hi:.2f}]"

    print("\nUPDATE time per run (seconds) (timing only training/update phases):")
    print(fmt_time("WalkSAT warm-start", ws_warm_times))
    print(fmt_time("WalkSAT restart/batch", ws_restart_times))
    print(fmt_time("WalkSAT epsilon-restart", ws_eps_times))
    print(fmt_time("ProbSAT warm-start", ps_warm_times))
    print(fmt_time("ProbSAT restart/batch", ps_restart_times))
    print(fmt_time("ProbSAT epsilon-restart", ps_eps_times))
    print(fmt_time("GA (best-of-pop)", ga_times))

    print("\nWilcoxon signed-rank tests (paired, two-sided) on per-run mean prequential satisfaction:")
    for name, baseline in [
        ("GA vs WalkSAT warm-start", ws_warm_means),
        ("GA vs WalkSAT restart/batch", ws_restart_means),
        ("GA vs WalkSAT epsilon-restart", ws_eps_means),
        ("GA vs ProbSAT warm-start", ps_warm_means),
        ("GA vs ProbSAT restart/batch", ps_restart_means),
        ("GA vs ProbSAT epsilon-restart", ps_eps_means),
    ]:
        res = wilcoxon_signed_rank_pvalue(ga_means, baseline)
        print(f"{name:<30} n={int(res['n']):2d}  W={res['W']:.2f}  z={res['z']:.3f}  p≈{res['p']:.4g}")

if __name__ == "__main__":
    main()
