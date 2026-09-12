"""
Scaling Probe: find the REAL resource wall for XGBoost, Random Forest, and
DeepCOMBI -- not MDR's combinatorial wall (already crossed at N=1000, order=3),
but the actual point where dense/tree-based methods themselves start failing
or slowing down dramatically.

This uses RANDOM (non-epistatic) data deliberately -- we are timing raw
computational cost, not detection accuracy. Detection accuracy at whatever
scale this reveals is a SEPARATE follow-up experiment (Step 2 in the plan).

Design notes for trustworthy output (per the verification lessons learned):
  - Every result is written to CSV immediately after each N (not buffered
    until the end), so a crash at N=300000 doesn't lose the N=1000..100000
    results.
  - Wall-clock time AND peak memory (via resource.getrusage on Linux, or a
    graceful fallback) are both recorded.
  - A hard per-method timeout (default 10 min) so one slow method doesn't
    block the whole sweep; a timeout is recorded as a real data point
    ("TIMEOUT"), not skipped silently.
  - Every row includes a run_id + timestamp, same discipline as the fair
    benchmark script, so stale/reused results are immediately detectable.
  - Prints go to stdout AND get written to the CSV -- if you're relaying
    output through an agent, paste the CSV content directly, not a summary.
"""

import argparse
import csv
import os
import time
import uuid
import signal
import platform
from datetime import datetime, timezone

import numpy as np


def _get_peak_memory_mb():
    """Best-effort peak RSS memory in MB. Uses psutil if available, then resource on Unix."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes -- normalize by platform
        if platform.system() == "Darwin":
            return peak_kb / (1024 * 1024)
        return peak_kb / 1024
    except ImportError:
        return None


import concurrent.futures

class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(fn, timeout_sec):
    """Runs fn() with a hard wall-clock timeout. Uses SIGALRM on Unix, ThreadPoolExecutor on Windows."""
    if platform.system() == "Windows":
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout_sec)
            executor.shutdown(wait=False)
            return result, False
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return None, True
        except Exception:
            executor.shutdown(wait=False)
            raise
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = fn()
        return result, False
    except TimeoutException:
        return None, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def probe_method(method_name, fit_fn, n_snps, n_samples, timeout_sec):
    print(f"  [{method_name}] N={n_snps} starting...", flush=True)
    t0 = time.time()

    def _run():
        np.random.seed(0)
        X = np.random.randint(0, 3, size=(n_samples, n_snps)).astype(np.float32)
        y = np.random.randint(0, 2, size=n_samples)
        fit_fn(X, y)
        return True

    try:
        result, timed_out = run_with_timeout(_run, timeout_sec)
    except MemoryError:
        elapsed = time.time() - t0
        print(f"  [{method_name}] N={n_snps} -> MemoryError after {elapsed:.1f}s", flush=True)
        return {"status": "MemoryError", "elapsed_sec": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{method_name}] N={n_snps} -> ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
        return {"status": f"ERROR:{type(e).__name__}", "elapsed_sec": round(elapsed, 1)}

    elapsed = time.time() - t0
    if timed_out:
        print(f"  [{method_name}] N={n_snps} -> TIMEOUT after {elapsed:.1f}s (limit {timeout_sec}s)", flush=True)
        return {"status": "TIMEOUT", "elapsed_sec": round(elapsed, 1)}

    peak_mem = _get_peak_memory_mb()
    mem_str = f"{peak_mem:.0f}MB" if peak_mem is not None else "N/A"
    print(f"  [{method_name}] N={n_snps} -> OK in {elapsed:.1f}s (peak mem: {mem_str})", flush=True)
    return {"status": "OK", "elapsed_sec": round(elapsed, 1), "peak_mem_mb": round(peak_mem, 0) if peak_mem else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1600)
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--scales", type=int, nargs="+",
                         default=[1000, 5000, 20000, 50000, 100000, 300000, 500000])
    parser.add_argument("--output", type=str, default="reports/benchmarks/scaling_probe.csv")
    args = parser.parse_args()

    from epistasis.models.baselines import TreeEnsembleBaseline
    from epistasis.models.deepcombi_mlp import DeepCOMBIBaseline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"=== Scaling Probe | run_id={run_id} | device={device} | "
          f"n_samples={args.n_samples} | timeout={args.timeout_sec}s ===", flush=True)
    print(f"NOTE: this script writes to {args.output} incrementally -- if relaying "
          f"through an agent, paste the raw CSV content directly, not a summary.", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)
    csv_file = open(args.output, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["run_id", "timestamp", "device", "n_samples", "n_snps",
                          "method", "status", "elapsed_sec", "peak_mem_mb"])

    methods = {
        "XGBoost": lambda X, y: TreeEnsembleBaseline(model_type="xgboost", n_estimators=100).fit(X, y),
        "RandomForest": lambda X, y: TreeEnsembleBaseline(model_type="random_forest", n_estimators=100).fit(X, y),
        "DeepCOMBI": None,  # constructed per-N below since it needs num_snps at init
    }

    for n_snps in args.scales:
        print(f"\n=== N_SNPS = {n_snps:,} ===", flush=True)
        methods["DeepCOMBI"] = lambda X, y, n=n_snps: DeepCOMBIBaseline(
            num_snps=n, epochs=10, device=device
        ).fit(X, y)  # fewer epochs here -- this is a resource probe, not an accuracy test

        any_ok = False
        for method_name, fit_fn in methods.items():
            res = probe_method(method_name, fit_fn, n_snps, args.n_samples, args.timeout_sec)
            writer.writerow([run_id, timestamp, device, args.n_samples, n_snps, method_name,
                              res["status"], res["elapsed_sec"], res.get("peak_mem_mb")])
            csv_file.flush()  # write immediately -- don't lose progress on a later crash
            if res["status"] == "OK":
                any_ok = True

        if not any_ok:
            print(f"\nAll methods failed/timed out at N={n_snps:,} -- stopping sweep here.", flush=True)
            break

    csv_file.close()
    print(f"\n=== DONE. Results appended to {args.output} ===", flush=True)
    print("To verify, run: sha256sum " + args.output, flush=True)


if __name__ == "__main__":
    main()
