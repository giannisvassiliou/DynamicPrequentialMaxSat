#!/usr/bin/env python3
"""
Numba is REQUIRED (no fallback).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange  # type: ignore
except Exception as e:
    raise SystemExit(
        "ERROR: Numba is required for this script but is not available.\n"
        "Install with: pip install numba\n"
        f"Original import error: {e}"
    )

# -----------------------------
# CNF flat
# -----------------------------
@dataclass(frozen=True)
class CNFFlat:
    n_vars: int
    lits: np.ndarray
    ptr: np.ndarray
    n_clauses: int

def load_cnf_flat(path: str) -> CNFFlat:
    clauses: List[List[int]] = []
    n_vars = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4:
                    n_vars = int(parts[2])
                continue
            lits = []
            for tok in line.split():
                v = int(tok)
                if v != 0:
                    lits.append(v)
            if lits:
                clauses.append(lits)

    if n_vars <= 0:
        mx = 0
        for cl in clauses:
            for lit in cl:
                v = abs(lit)
                if v > mx:
                    mx = v
        n_vars = mx

    flat: List[int] = []
    ptr = [0]
    for cl in clauses:
        flat.extend(cl)
        ptr.append(len(flat))

    return CNFFlat(
        n_vars=n_vars,
        lits=np.asarray(flat, dtype=np.int32),
        ptr=np.asarray(ptr, dtype=np.int32),
        n_clauses=len(clauses),
    )

# -----------------------------
# Numba scoring kernels (uint8)
# -----------------------------
@njit(cache=True)
def score_assignment_subset_u8(lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray, clause_idx: np.ndarray) -> int:
    sat = 0
    for k in range(clause_idx.shape[0]):
        c = clause_idx[k]
        start = ptr[c]
        end = ptr[c + 1]
        for i in range(start, end):
            lit = lits[i]
            v = abs(lit) - 1
            av = a_u8[v]
            if (lit > 0 and av == 1) or (lit < 0 and av == 0):
                sat += 1
                break
    return sat

@njit(cache=True)
def score_pop_serial_u8(lits: np.ndarray, ptr: np.ndarray, pop_u8: np.ndarray, clause_idx: np.ndarray) -> np.ndarray:
    out = np.empty(pop_u8.shape[0], dtype=np.int32)
    for i in range(pop_u8.shape[0]):
        out[i] = score_assignment_subset_u8(lits, ptr, pop_u8[i], clause_idx)
    return out

@njit(cache=True, parallel=True)
def score_pop_parallel_u8(lits: np.ndarray, ptr: np.ndarray, pop_u8: np.ndarray, clause_idx: np.ndarray) -> np.ndarray:
    out = np.empty(pop_u8.shape[0], dtype=np.int32)
    for i in prange(pop_u8.shape[0]):
        out[i] = score_assignment_subset_u8(lits, ptr, pop_u8[i], clause_idx)
    return out

# -----------------------------
# GA next generation in Numba using np.random
# -----------------------------
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

    # Numba supports np.argsort
    order = np.argsort(fits)[::-1]
    new_pop = np.empty_like(pop)

    # elites
    for i in range(elite_n):
        src = order[i]
        for v in range(V):
            new_pop[i, v] = pop[src, v]

    # children
    for row in range(elite_n, P):
        p1 = tournament_pick(fits, tournament_k)
        p2 = tournament_pick(fits, tournament_k)

        if np.random.random() < crossover_rate:
            cut = np.random.randint(1, V)  # [1, V-1]
            for v in range(cut):
                new_pop[row, v] = pop[p1, v]
            for v in range(cut, V):
                new_pop[row, v] = pop[p2, v]
        else:
            for v in range(V):
                new_pop[row, v] = pop[p1, v]

        # mutation
        for v in range(V):
            if np.random.random() < mutation_rate:
                new_pop[row, v] ^= 1

    return new_pop

# -----------------------------
# WalkSAT (FULLY NUMBA, parity-style)
# -----------------------------
@dataclass
class WalkSATConfig:
    flips: int = 2000
    p_random: float = 0.55
    # random unsat-clause probes before full scan (speed)
    unsat_probe_tries: int = 16
    # number of candidate literals sampled for greedy "best" step
    best_lit_samples: int = 5

@njit(cache=True)
def clause_satisfied_flat_u8_ws(c: int, lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray) -> bool:
    start = ptr[c]
    end = ptr[c + 1]
    for i in range(start, end):
        lit = lits[i]
        v = abs(lit) - 1
        av = a_u8[v]
        if (lit > 0 and av == 1) or (lit < 0 and av == 0):
            return True
    return False

@njit(cache=True)
def find_unsat_clause_subset_ws(lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray, clause_idx: np.ndarray, probe_tries: int) -> int:
    m = clause_idx.shape[0]
    for _ in range(probe_tries):
        pos = np.random.randint(0, m)
        c = int(clause_idx[pos])
        if not clause_satisfied_flat_u8_ws(c, lits, ptr, a_u8):
            return c
    for pos in range(m):
        c = int(clause_idx[pos])
        if not clause_satisfied_flat_u8_ws(c, lits, ptr, a_u8):
            return c
    return -1

@njit(cache=True)
def walksat_one_restart_numba(a0_u8: np.ndarray, lits: np.ndarray, ptr: np.ndarray, clause_idx: np.ndarray,
                             flips: int, p_random: float, probe_tries: int, best_lit_samples: int) -> np.ndarray:
    a = a0_u8.copy()
    m = clause_idx.shape[0]
    if m <= 0:
        return a

    for _ in range(flips):
        c = find_unsat_clause_subset_ws(lits, ptr, a, clause_idx, probe_tries)
        if c < 0:
            break  # all satisfied

        start = ptr[c]
        end = ptr[c + 1]
        k = end - start
        if k <= 0:
            continue

        # random or greedy choice
        if np.random.random() < p_random:
            lit = lits[start + np.random.randint(0, k)]
            v = abs(lit) - 1
        else:
            # sample a few literals from the clause (with replacement) and pick best by score
            # (mirrors your Python version's "best-of-k" sample; ties resolved by first-best)
            samples = best_lit_samples if best_lit_samples < k else k
            best_v = abs(lits[start]) - 1
            best_sc = -1
            for _s in range(samples):
                lit = lits[start + np.random.randint(0, k)]
                v2 = abs(lit) - 1
                if v2 < 0 or v2 >= a.shape[0]:
                    continue
                a[v2] ^= 1
                sc = score_assignment_subset_u8(lits, ptr, a, clause_idx)
                a[v2] ^= 1
                if sc > best_sc:
                    best_sc = sc
                    best_v = v2
            v = best_v

        if 0 <= v < a.shape[0]:
            a[v] ^= 1

    return a

def walksat_update_u8(a_u8: np.ndarray, cnf: CNFFlat, clause_idx: np.ndarray, cfg: WalkSATConfig) -> np.ndarray:
    # Single restart (consistent with prior WalkSAT block); caller controls restarts/epsilon behavior.
    return walksat_one_restart_numba(
        a_u8, cnf.lits, cnf.ptr, clause_idx,
        cfg.flips, cfg.p_random, cfg.unsat_probe_tries, cfg.best_lit_samples
    )
# -----------------------------
# ProbSAT (NUMBA, canonical break-based)
# -----------------------------
@dataclass
class ProbSATConfig:
    flips: int = 2000
    cb: float = 2.06
    # random unsat-clause probes before full scan (speed)
    unsat_probe_tries: int = 32

@njit(cache=True)
def clause_satisfied_flat_u8(c: int, lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray) -> bool:
    start = ptr[c]
    end = ptr[c + 1]
    for i in range(start, end):
        lit = lits[i]
        v = abs(lit) - 1
        av = a_u8[v]
        if (lit > 0 and av == 1) or (lit < 0 and av == 0):
            return True
    return False

@njit(cache=True)
def find_unsat_clause_pos_subset(lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray, clause_idx: np.ndarray, probe_tries: int) -> int:
    m = clause_idx.shape[0]
    # probe random subset positions
    for _ in range(probe_tries):
        pos = np.random.randint(0, m)
        c = int(clause_idx[pos])
        if not clause_satisfied_flat_u8(c, lits, ptr, a_u8):
            return pos
    # full scan
    for pos in range(m):
        c = int(clause_idx[pos])
        if not clause_satisfied_flat_u8(c, lits, ptr, a_u8):
            return pos
    return -1

@njit(cache=True)
def probsat_break_count_var_subset(lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray, clause_idx: np.ndarray, v: int) -> int:
    # Count clauses that are currently SAT and would become UNSAT if v is flipped (break count)
    brk = 0
    for kk in range(clause_idx.shape[0]):
        c = int(clause_idx[kk])
        start = ptr[c]
        end = ptr[c + 1]
        sat_count = 0
        sat_by_v = 0
        for i in range(start, end):
            lit = lits[i]
            var = abs(lit) - 1
            av = a_u8[var]
            is_sat = (lit > 0 and av == 1) or (lit < 0 and av == 0)
            if is_sat:
                sat_count += 1
                if var == v:
                    sat_by_v += 1
        if sat_by_v > 0 and sat_count == sat_by_v:
            brk += 1
    return brk

@njit(cache=True)
def probsat_pick_var_from_clause(c: int, lits: np.ndarray, ptr: np.ndarray, a_u8: np.ndarray, clause_idx: np.ndarray, cb: float) -> int:
    start = ptr[c]
    end = ptr[c + 1]
    k = end - start
    if k <= 0:
        return 0

    weights = np.empty(k, dtype=np.float64)
    # compute weights for each literal's variable in the clause
    for j in range(k):
        lit = lits[start + j]
        v = abs(lit) - 1
        brk = probsat_break_count_var_subset(lits, ptr, a_u8, clause_idx, v)
        weights[j] = 1.0 / ((brk + 1.0) ** cb)

    s = 0.0
    for j in range(k):
        s += weights[j]
    if s <= 0.0:
        lit = lits[start + np.random.randint(0, k)]
        return abs(lit) - 1

    r = np.random.random() * s
    acc = 0.0
    for j in range(k):
        acc += weights[j]
        if acc >= r:
            lit = lits[start + j]
            return abs(lit) - 1

    lit = lits[start]
    return abs(lit) - 1

@njit(cache=True)
def probsat_one_restart_numba(a0_u8: np.ndarray, lits: np.ndarray, ptr: np.ndarray, clause_idx: np.ndarray,
                             flips: int, cb: float, probe_tries: int) -> np.ndarray:
    a = a0_u8.copy()
    for _ in range(flips):
        pos = find_unsat_clause_pos_subset(lits, ptr, a, clause_idx, probe_tries)
        if pos < 0:
            break
        c = int(clause_idx[pos])
        v = probsat_pick_var_from_clause(c, lits, ptr, a, clause_idx, cb)
        if 0 <= v < a.shape[0]:
            a[v] ^= 1
    return a

def probsat_update_u8(a_u8: np.ndarray, cnf: CNFFlat, clause_idx: np.ndarray, cfg: ProbSATConfig) -> np.ndarray:
    # Single restart (consistent with WalkSAT block); caller controls restarts/epsilon behavior.
    return probsat_one_restart_numba(a_u8, cnf.lits, cnf.ptr, clause_idx, cfg.flips, cfg.cb, cfg.unsat_probe_tries)


# -----------------------------
# Stats helpers
# -----------------------------
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
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
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

# -----------------------------
# One run — same outputs
# -----------------------------
@dataclass
class RunResult:
    ws_warm: float
    ws_restart: float
    ws_eps: float
    ps_warm: float
    ps_restart: float
    ps_eps: float
    ga_best: float
    t_ws_warm: float
    t_ws_restart: float
    t_ws_eps: float
    t_ps_warm: float
    t_ps_restart: float
    t_ps_eps: float
    t_ga: float

def rand_u8_vec(n: int) -> np.ndarray:
    return (np.random.rand(n) < 0.5).astype(np.uint8)

def run_one(seed: int, stream: List[CNFFlat], epochs: int, max_vars: int, batch_size: Optional[int],
            normalize: bool, epsilon: float, ws_cfg: WalkSATConfig, ps_cfg: ProbSATConfig, pop_size: int, elite_frac: float,
            mutation_rate: float, tournament_k: int, generations_per_step: int, crossover_rate: float,
            progress: bool, numba_parallel: bool) -> RunResult:
    random.seed(seed)
    np.random.seed(seed)

    ws_warm = rand_u8_vec(max_vars)
    ws_eps = rand_u8_vec(max_vars)
    ps_warm = rand_u8_vec(max_vars)
    ps_eps = rand_u8_vec(max_vars)
    pop = (np.random.rand(pop_size, max_vars) < 0.5).astype(np.uint8)

    total_steps = epochs * len(stream)
    sum_ws_warm = sum_ws_restart = sum_ws_eps = sum_ps_warm = sum_ps_restart = sum_ps_eps = sum_ga = 0.0
    t_ws_warm = t_ws_restart = t_ws_eps = t_ps_warm = t_ps_restart = t_ps_eps = t_ga = 0.0
    step_idx = 0

    for ep in range(epochs):
        for cnf in stream:
            step_idx += 1
            if batch_size is not None and cnf.n_clauses > batch_size:
                clause_idx = np.random.choice(cnf.n_clauses, batch_size, replace=False).astype(np.int64)
            else:
                clause_idx = np.arange(cnf.n_clauses, dtype=np.int64)

            denom = float(clause_idx.shape[0]) if normalize else 1.0
            if denom <= 0:
                denom = 1.0

            # prequential test
            sum_ws_warm += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ws_warm[:cnf.n_vars], clause_idx)) / denom

            ws_restart = rand_u8_vec(max_vars)
            sum_ws_restart += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ws_restart[:cnf.n_vars], clause_idx)) / denom

            sum_ws_eps += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ws_eps[:cnf.n_vars], clause_idx)) / denom

            sum_ps_warm += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ps_warm[:cnf.n_vars], clause_idx)) / denom
            ps_restart = rand_u8_vec(max_vars)
            sum_ps_restart += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ps_restart[:cnf.n_vars], clause_idx)) / denom
            sum_ps_eps += float(score_assignment_subset_u8(cnf.lits, cnf.ptr, ps_eps[:cnf.n_vars], clause_idx)) / denom

            pop_view = pop[:, :cnf.n_vars]
            fits = score_pop_parallel_u8(cnf.lits, cnf.ptr, pop_view, clause_idx) if numba_parallel else score_pop_serial_u8(cnf.lits, cnf.ptr, pop_view, clause_idx)
            sum_ga += float(int(fits.max()) if fits.size else 0) / denom

            # update
            t0 = time.perf_counter()
            ws_warm[:cnf.n_vars] = walksat_update_u8(ws_warm[:cnf.n_vars].copy(), cnf, clause_idx, ws_cfg)
            t_ws_warm += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = walksat_update_u8(ws_restart[:cnf.n_vars].copy(), cnf, clause_idx, ws_cfg)
            t_ws_restart += time.perf_counter() - t0

            if random.random() < epsilon:
                ws_eps = rand_u8_vec(max_vars)
            t0 = time.perf_counter()
            ws_eps[:cnf.n_vars] = walksat_update_u8(ws_eps[:cnf.n_vars].copy(), cnf, clause_idx, ws_cfg)
            t_ws_eps += time.perf_counter() - t0

            t0 = time.perf_counter()
            ps_warm[:cnf.n_vars] = probsat_update_u8(ps_warm[:cnf.n_vars].copy(), cnf, clause_idx, ps_cfg)
            t_ps_warm += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = probsat_update_u8(ps_restart[:cnf.n_vars].copy(), cnf, clause_idx, ps_cfg)
            t_ps_restart += time.perf_counter() - t0

            if random.random() < epsilon:
                ps_eps = rand_u8_vec(max_vars)
            t0 = time.perf_counter()
            ps_eps[:cnf.n_vars] = probsat_update_u8(ps_eps[:cnf.n_vars].copy(), cnf, clause_idx, ps_cfg)
            t_ps_eps += time.perf_counter() - t0

            elite_n = max(1, int(round(elite_frac * pop_size)))
            t0 = time.perf_counter()
            for _ in range(generations_per_step):
                pop_view = pop[:, :cnf.n_vars]
                fits = score_pop_parallel_u8(cnf.lits, cnf.ptr, pop_view, clause_idx) if numba_parallel else score_pop_serial_u8(cnf.lits, cnf.ptr, pop_view, clause_idx)
                new_view = ga_next_generation_rng(pop_view, fits.astype(np.int32), elite_n, tournament_k, crossover_rate, mutation_rate)
                pop[:, :cnf.n_vars] = new_view
            t_ga += time.perf_counter() - t0

            if progress:
                print(f"  step {step_idx}/{total_steps} (epoch {ep+1}/{epochs}) done")

    return RunResult(
        ws_warm=sum_ws_warm / total_steps,
        ws_restart=sum_ws_restart / total_steps,
        ws_eps=sum_ws_eps / total_steps,
        ps_warm=sum_ps_warm / total_steps,
        ps_restart=sum_ps_restart / total_steps,
        ps_eps=sum_ps_eps / total_steps,
        ga_best=sum_ga / total_steps,
        t_ws_warm=t_ws_warm,
        t_ws_restart=t_ws_restart,
        t_ws_eps=t_ws_eps,
        t_ps_warm=t_ps_warm,
        t_ps_restart=t_ps_restart,
        t_ps_eps=t_ps_eps,
        t_ga=t_ga,
    )

# -----------------------------
# MAIN (same blocks)
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf_dir", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--numba_parallel", action="store_true")

    ap.add_argument("--walksat_flips", type=int, default=2000)
    ap.add_argument("--walksat_p_random", type=float, default=0.55)

    ap.add_argument("--probsat_flips", type=int, default=2000)
    ap.add_argument("--probsat_cb", type=float, default=2.06)
    ap.add_argument("--probsat_unsat_probe_tries", type=int, default=32)

    ap.add_argument("--pop_size", type=int, default=80)
    ap.add_argument("--elite_frac", type=float, default=0.15)
    ap.add_argument("--mutation_rate", type=float, default=0.01)
    ap.add_argument("--tournament_k", type=int, default=3)
    ap.add_argument("--ga_gens_per_step", type=int, default=4)
    ap.add_argument("--crossover_rate", type=float, default=0.9)

    args = ap.parse_args()

    cnf_files = sorted(glob.glob(os.path.join(args.cnf_dir, "*.cnf")))
    if not cnf_files:
        raise SystemExit("No .cnf files found in --cnf_dir")

    stream: List[CNFFlat] = []
    max_vars = 0
    for fp in cnf_files:
        cnf = load_cnf_flat(fp)
        max_vars = max(max_vars, cnf.n_vars)
        stream.append(cnf)

    total_steps = len(stream) * args.epochs
    print("Loaded {} CNF versions | max vars = {}".format(len(stream), max_vars))
    print("Replaying stream for {} epochs (total steps = {})".format(args.epochs, total_steps))
    if args.batch_size is not None:
        print("Using clause subsampling per step: batch_size={}".format(args.batch_size))
    print("Scoring mode: {}".format("normalized" if args.normalize else "raw clause counts"))
    print("Numba: ENABLED (required) — compiling kernels on first use")
    if args.numba_parallel:
        print("Numba parallel: ON (population scoring uses prange)")
    print("")

    ws_cfg = WalkSATConfig(flips=args.walksat_flips, p_random=args.walksat_p_random)
    ps_cfg = ProbSATConfig(flips=args.probsat_flips, cb=args.probsat_cb, unsat_probe_tries=args.probsat_unsat_probe_tries)

    # warmup compile
    dummy = stream[0]
    clause_idx0 = np.arange(min(dummy.n_clauses, 10), dtype=np.int64)
    a0 = (np.random.rand(dummy.n_vars) < 0.5).astype(np.uint8)
    _ = score_assignment_subset_u8(dummy.lits, dummy.ptr, a0, clause_idx0)
    pop0 = (np.random.rand(min(args.pop_size, 8), dummy.n_vars) < 0.5).astype(np.uint8)
    _ = score_pop_serial_u8(dummy.lits, dummy.ptr, pop0, clause_idx0)
    _ = ga_next_generation_rng(pop0, np.ones(pop0.shape[0], dtype=np.int32), max(1, int(round(args.elite_frac*min(args.pop_size,8)))), args.tournament_k, args.crossover_rate, args.mutation_rate)
    _ = probsat_one_restart_numba(a0, dummy.lits, dummy.ptr, clause_idx0, 1, args.probsat_cb, args.probsat_unsat_probe_tries)

    ws_warm_vals: List[float] = []
    ws_restart_vals: List[float] = []
    ws_eps_vals: List[float] = []
    ps_warm_vals: List[float] = []
    ps_restart_vals: List[float] = []
    ps_eps_vals: List[float] = []
    ga_vals: List[float] = []

    t_ws_warm_vals: List[float] = []
    t_ws_restart_vals: List[float] = []
    t_ws_eps_vals: List[float] = []
    t_ps_warm_vals: List[float] = []
    t_ps_restart_vals: List[float] = []
    t_ps_eps_vals: List[float] = []
    t_ga_vals: List[float] = []

    for i in range(args.runs):
        seed = args.seed0 + i
        rr = run_one(
            seed, stream, args.epochs, max_vars, args.batch_size, args.normalize, args.epsilon,
            ws_cfg, ps_cfg, args.pop_size, args.elite_frac, args.mutation_rate, args.tournament_k,
            args.ga_gens_per_step, args.crossover_rate, args.progress, args.numba_parallel
        )
        ws_warm_vals.append(rr.ws_warm)
        ws_restart_vals.append(rr.ws_restart)
        ws_eps_vals.append(rr.ws_eps)
        ps_warm_vals.append(rr.ps_warm)
        ps_restart_vals.append(rr.ps_restart)
        ps_eps_vals.append(rr.ps_eps)
        ga_vals.append(rr.ga_best)
        t_ws_warm_vals.append(rr.t_ws_warm)
        t_ws_restart_vals.append(rr.t_ws_restart)
        t_ws_eps_vals.append(rr.t_ws_eps)
        t_ps_warm_vals.append(rr.t_ps_warm)
        t_ps_restart_vals.append(rr.t_ps_restart)
        t_ps_eps_vals.append(rr.t_ps_eps)
        t_ga_vals.append(rr.t_ga)

        print("RUN {:02d} seed={}  WS_warm={:.6f}  WS_restart={:.6f}  WS_eps={:.6f}  PS_warm={:.6f}  PS_restart={:.6f}  PS_eps={:.6f}  GA={:.6f}  t_WS={:.2f}s  t_GA={:.2f}s".format(
            i, seed, rr.ws_warm, rr.ws_restart, rr.ws_eps, rr.ps_warm, rr.ps_restart, rr.ps_eps, rr.ga_best, rr.t_ws_warm, rr.t_ga
        ))

    print("")
    print("Protocol: averaged over {} runs. Each run has {} steps ({} CNFs × {} epochs).".format(args.runs, total_steps, len(stream), args.epochs))
    print("Epsilon (WalkSAT-eps) = {}".format(args.epsilon))
    print("")

    def fmt_stats(name: str, xs: List[float]) -> str:
        m = mean(xs); s = stdev(xs); lo, hi = ci95(xs)
        return "{:<24} mean={:.6f}  std={:.6f}  95%CI=[{:.6f},{:.6f}]".format(name, m, s, lo, hi)

    print("Prequential score (per-run mean over steps):")
    print(fmt_stats("WalkSAT warm-start", ws_warm_vals))
    print(fmt_stats("WalkSAT restart/step", ws_restart_vals))
    print(fmt_stats("WalkSAT epsilon-restart", ws_eps_vals))
    print(fmt_stats("ProbSAT warm-start", ps_warm_vals))
    print(fmt_stats("ProbSAT restart/step", ps_restart_vals))
    print(fmt_stats("ProbSAT epsilon-restart", ps_eps_vals))
    print(fmt_stats("GA (best-of-pop)", ga_vals))
    print("")

    def fmt_time(name: str, xs: List[float]) -> str:
        m = mean(xs); s = stdev(xs); lo, hi = ci95(xs)
        return "{:<24} mean={:.3f}s  std={:.3f}s  95%CI=[{:.3f},{:.3f}]".format(name, m, s, lo, hi)

    print("UPDATE time per run (seconds; timed only update phases):")
    print(fmt_time("WalkSAT warm-start", t_ws_warm_vals))
    print(fmt_time("WalkSAT restart/step", t_ws_restart_vals))
    print(fmt_time("WalkSAT epsilon-restart", t_ws_eps_vals))
    print(fmt_time("ProbSAT warm-start", t_ps_warm_vals))
    print(fmt_time("ProbSAT restart/step", t_ps_restart_vals))
    print(fmt_time("ProbSAT epsilon-restart", t_ps_eps_vals))
    print(fmt_time("GA (best-of-pop)", t_ga_vals))
    print("")

    print("Wilcoxon signed-rank tests (paired, two-sided) on per-run mean scores:")
    for label, baseline in [
        ("GA vs WalkSAT warm-start", ws_warm_vals),
        ("GA vs WalkSAT restart/step", ws_restart_vals),
        ("GA vs WalkSAT epsilon-restart", ws_eps_vals),
        ("GA vs ProbSAT warm-start", ps_warm_vals),
        ("GA vs ProbSAT restart/step", ps_restart_vals),
        ("GA vs ProbSAT epsilon-restart", ps_eps_vals),
    ]:
        res = wilcoxon_signed_rank_pvalue(ga_vals, baseline)
        print("{:<28} n={}  W={:.2f}  z={:.3f}  p≈{:.4g}".format(label, int(res["n"]), res["W"], res["z"], res["p"]))

if __name__ == "__main__":
    main()
