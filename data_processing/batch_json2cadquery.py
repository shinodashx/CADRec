import argparse
from collections import deque
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from tqdm import tqdm
# 9185

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert DeepCAD JSON files to CadQuery code with STL export."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent / "cadmllm",
        help="Root containing four-digit subfolders with source JSON files (default: data_processing/cadmllm).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "cadmllmnew",
        help="Root where outputs will be written (default: data_processing/cadmllmnew).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalization to [-100, 100] range.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied after normalization (default: 1.0).",
    )
    parser.add_argument(
        "--folders",
        type=str,
        default=None,
        help="Comma-separated list of folder names to process (e.g., '0000,0001'). Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of parallel workers (default: 24).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each JSON file (default: 300).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Limit math/runtime threads inside each worker. Use 1 to avoid oversubscription.",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default="auto",
        choices=["auto", "spawn", "fork", "forkserver"],
        help="Multiprocessing start method. auto prefers fork on Linux and spawn elsewhere.",
    )
    parser.add_argument(
        "--start-retries",
        type=int,
        default=10,
        help="How many times to requeue a file if worker process creation fails.",
    )
    return parser.parse_args()


def _iter_subfolders(root: Path, folder_filter: Optional[list[str]] = None) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root {root} does not exist.")
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and len(path.name) == 4
    )
    if folder_filter:
        candidates = [p for p in candidates if p.name in folder_filter]
    return candidates


def _terminate_process(proc: mp.Process) -> None:
    if not proc.is_alive():
        return
    try:
        proc.kill()
    except AttributeError:
        proc.terminate()


def _apply_thread_env(threads_per_worker: int) -> None:
    if threads_per_worker <= 0:
        return
    value = str(int(threads_per_worker))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TBB_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = value
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ.setdefault("KMP_BLOCKTIME", "0")


def _configure_worker_runtime(threads_per_worker: int) -> None:
    _apply_thread_env(threads_per_worker)
    try:
        import torch

        torch.set_num_threads(max(1, int(threads_per_worker)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _resolve_start_method(name: str) -> str:
    if name != "auto":
        return name
    if sys.platform.startswith("linux"):
        return "fork"
    return "spawn"


def _worker_exit_error(proc: mp.Process, detail: Optional[str] = None) -> str:
    exitcode = proc.exitcode
    if detail:
        return f"{detail} (exitcode={exitcode})"
    if exitcode is None:
        return "worker exited without result"
    if exitcode < 0:
        return f"worker terminated by signal {-exitcode}"
    return f"worker exited without result (exitcode={exitcode})"


def _safe_recv_result(conn, proc: mp.Process):
    try:
        return True, conn.recv()
    except EOFError:
        return False, _worker_exit_error(proc, "worker pipe closed before sending result")
    except OSError as exc:
        return False, _worker_exit_error(proc, f"worker pipe recv failed: {exc}")


def validate_output(output_dir: Path) -> tuple[bool, str]:
    """
    Validate that the output is consistent.
    Returns (is_valid, error_message)
    """
    def _validate_part_like_dir(part_dir: Path) -> tuple[bool, str]:
        part_py = part_dir / f"{part_dir.name}.py"
        part_stl = part_dir / f"{part_dir.name}.stl"
        part_ply = part_dir / f"{part_dir.name}.ply"
        part_npy = part_dir / f"{part_dir.name}.npy"
        part_crop_ply = part_dir / f"{part_dir.name}_crop.ply"
        part_crop_npy = part_dir / f"{part_dir.name}_crop.npy"
        part_bbox = part_dir / "bbox.json"
        if not part_py.exists():
            return False, f"Missing part py: {part_py}"
        if not part_stl.exists():
            return False, f"Missing part STL: {part_stl}"
        if not part_ply.exists():
            return False, f"Missing part PLY: {part_ply}"
        if not part_npy.exists():
            return False, f"Missing part NPY: {part_npy}"
        if not part_crop_ply.exists():
            return False, f"Missing part crop PLY: {part_crop_ply}"
        if not part_crop_npy.exists():
            return False, f"Missing part crop NPY: {part_crop_npy}"
        if not part_bbox.exists():
            return False, f"Missing part bbox.json: {part_bbox}"
        try:
            with open(part_bbox, "r", encoding="utf-8") as f:
                part_bbox_data = json.load(f)
            label = part_bbox_data.get("label")
            model_bbox = part_bbox_data.get("model_bbox")
            steps = part_bbox_data.get("steps", {})
            if label != "STOP":
                return False, f"Part bbox.json label should be STOP: {part_bbox}"
            if not isinstance(model_bbox, list) or len(model_bbox) != 6:
                return False, f"Invalid part model_bbox: {part_bbox}"
            if not isinstance(steps, dict):
                return False, f"Invalid part steps: {part_bbox}"
        except Exception as e:
            return False, f"Failed to read part bbox.json {part_bbox}: {e}"
        return True, ""

    bbox_path = output_dir / "bbox.json"

    if not bbox_path.exists():
        return False, "bbox.json missing"

    try:
        with open(bbox_path, "r", encoding="utf-8") as f:
            bbox_data = json.load(f)
        label = bbox_data.get("label")
        model_bbox = bbox_data.get("model_bbox")
        if label == "STOP":
            return _validate_part_like_dir(output_dir)
        if label != "SPLIT":
            return False, f"Root bbox.json label should be SPLIT or STOP, got {label!r}"
        parts = bbox_data.get("parts", {})
        if not isinstance(model_bbox, list) or len(model_bbox) != 6:
            return False, "Root bbox.json missing valid model_bbox"
        if not isinstance(parts, dict):
            return False, "Root bbox.json missing valid parts"
        bbox_part_count = len(parts)
    except Exception as e:
        return False, f"Failed to read bbox.json: {e}"

    model_stl = output_dir / f"{output_dir.name}.stl"
    if not model_stl.exists():
        return False, f"Model STL {model_stl.name} missing"

    model_ply = output_dir / f"{output_dir.name}.ply"
    if not model_ply.exists():
        return False, f"Model PLY {model_ply.name} missing"

    model_npy = output_dir / f"{output_dir.name}.npy"
    if not model_npy.exists():
        return False, f"Model NPY {model_npy.name} missing"

    model_py = output_dir / f"{output_dir.name}.py"
    if not model_py.exists():
        return False, f"Model py {model_py.name} missing"

    sibling_part_dirs = sorted(
        path for path in output_dir.parent.iterdir()
        if path.is_dir() and re.fullmatch(rf"{re.escape(output_dir.name)}_\d+", path.name)
    )
    if len(sibling_part_dirs) != bbox_part_count:
        return False, (
            f"sibling part dir count ({len(sibling_part_dirs)}) "
            f"!= bbox parts count ({bbox_part_count})"
        )

    for part_dir in sibling_part_dirs:
        is_valid, error_msg = _validate_part_like_dir(part_dir)
        if not is_valid:
            return is_valid, error_msg

    return True, ""


def _json_worker(
    json_path_str: str,
    output_dir_str: str,
    normalize: bool,
    scale: float,
    threads_per_worker: int,
    result_conn,
) -> None:
    """Worker function that runs in a separate process."""
    _configure_worker_runtime(threads_per_worker)
    from json2cadquery import generate_full_output

    json_path = Path(json_path_str)
    output_dir = Path(output_dir_str)

    try:
        generate_full_output(
            str(json_path),
            str(output_dir),
            normalize=normalize,
            scale=scale,
            quiet=True,
        )

        is_valid, error_msg = validate_output(output_dir)
        if is_valid:
            result_conn.send(("ok", json_path_str, ""))
        else:
            result_conn.send(("fail", json_path_str, error_msg))

    except Exception:
        result_conn.send(("fail", json_path_str, traceback.format_exc()))
    finally:
        result_conn.close()


def _run_json_safe(
    json_path: Path,
    output_dir: Path,
    normalize: bool,
    scale: float,
    timeout: float,
    threads_per_worker: int = 1,
    start_method: str = "auto",
) -> tuple[bool, str]:
    """Run JSON processing in isolated process with timeout."""
    ctx = mp.get_context(_resolve_start_method(start_method))
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_json_worker,
        args=(
            str(json_path),
            str(output_dir),
            normalize,
            scale,
            threads_per_worker,
            child_conn,
        ),
    )
    proc.start()
    child_conn.close()

    result = None
    error: Optional[str] = None

    try:
        if timeout > 0:
            if parent_conn.poll(timeout):
                recv_ok, recv_value = _safe_recv_result(parent_conn, proc)
                if recv_ok:
                    result = recv_value
                else:
                    error = recv_value
            else:
                if proc.is_alive():
                    error = f"timed out after {timeout} seconds"
                else:
                    error = _worker_exit_error(proc)
        else:
            proc.join()
            if parent_conn.poll():
                recv_ok, recv_value = _safe_recv_result(parent_conn, proc)
                if recv_ok:
                    result = recv_value
                else:
                    error = recv_value
            else:
                error = _worker_exit_error(proc)
    finally:
        if proc.is_alive():
            _terminate_process(proc)
        proc.join()
        parent_conn.close()

    if error:
        return False, error

    if result is None:
        return False, "no result from worker"

    status, _, msg = result
    if status == "ok":
        return True, ""
    return False, msg


def main() -> None:
    args = parse_args()

    folder_filter = None
    if args.folders:
        folder_filter = [f.strip() for f in args.folders.split(",")]

    subfolders = _iter_subfolders(args.input_root, folder_filter)
    if not subfolders:
        raise SystemExit(f"No subfolders found under {args.input_root}")

    # Collect all JSON files
    all_json_files: list[tuple[Path, Path]] = []  # (json_path, output_dir)
    for folder in subfolders:
        json_files = sorted(folder.glob("*.json"))
        for json_path in json_files:
            json_name = json_path.stem
            subfolder = json_path.parent.name
            output_dir = args.output_root / subfolder / json_name
            all_json_files.append((json_path, output_dir))

    normalize = not args.no_normalize
    scale = args.scale
    timeout = args.timeout
    worker_count = max(1, args.workers)
    threads_per_worker = max(1, args.threads_per_worker)
    start_retries = max(0, args.start_retries)

    total_files = len(all_json_files)
    print(f"[batch] Found {total_files} JSON files to process")
    print(f"[batch] Output root: {args.output_root}")
    print(f"[batch] Workers: {args.workers}, Timeout: {args.timeout}s")
    print(f"[batch] Threads/worker: {args.threads_per_worker}, Start method: {_resolve_start_method(args.start_method)}")
    print(f"[batch] Start retries: {start_retries}")
    print(f"[batch] Normalize: {not args.no_normalize}, Scale: {args.scale}")
    print()

    total_success = 0
    total_fail = 0
    all_failures = []
    total_requeued = 0

    _apply_thread_env(threads_per_worker)
    ctx = mp.get_context(_resolve_start_method(args.start_method))
    pending = deque((json_path, output_dir, 0) for json_path, output_dir in all_json_files)
    # Track active processes: list of (json_path, output_dir, proc, conn, start_time)
    active = []

    pbar = tqdm(total=total_files, desc="Processing", unit="file")

    while pending or active:
        start_failed_this_round = False
        # Start new workers
        while pending and len(active) < worker_count:
            json_path, output_dir, start_attempt = pending.popleft()
            output_dir.mkdir(parents=True, exist_ok=True)

            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_json_worker,
                args=(
                    str(json_path),
                    str(output_dir),
                    normalize,
                    scale,
                    threads_per_worker,
                    child_conn,
                ),
            )
            try:
                proc.start()
            except (BlockingIOError, OSError) as exc:
                parent_conn.close()
                child_conn.close()
                start_failed_this_round = True
                if start_attempt < start_retries:
                    pending.append((json_path, output_dir, start_attempt + 1))
                    total_requeued += 1
                    pbar.set_postfix({
                        "ok": total_success,
                        "fail": total_fail,
                        "retry": total_requeued,
                        "last": f"{json_path.parent.name}/{json_path.name} REQUEUE",
                    })
                else:
                    total_fail += 1
                    all_failures.append({
                        "file": str(json_path),
                        "error": f"worker start failed after {start_attempt + 1} attempts: {exc}",
                    })
                    pbar.set_postfix({
                        "ok": total_success,
                        "fail": total_fail,
                        "retry": total_requeued,
                        "last": f"{json_path.parent.name}/{json_path.name} STARTFAIL",
                    })
                    pbar.update(1)
                break
            child_conn.close()
            active.append((json_path, output_dir, proc, parent_conn, time.time()))

        # Check all active tasks for completion or timeout
        if active:
            still_active = []
            for json_path, output_dir, proc, result_conn, start_time in active:
                subfolder = json_path.parent.name
                elapsed = time.time() - start_time

                # Check if timed out
                if elapsed > timeout:
                    if proc.is_alive():
                        _terminate_process(proc)
                    proc.join(timeout=1)
                    result_conn.close()

                    total_fail += 1
                    all_failures.append({"file": str(json_path), "error": f"timed out after {timeout} seconds"})
                    pbar.set_postfix({"ok": total_success, "fail": total_fail, "retry": total_requeued, "last": f"{subfolder}/{json_path.name} TIMEOUT"})
                    pbar.update(1)
                    continue

                # Check if result available (non-blocking)
                if result_conn.poll():
                    recv_ok, recv_value = _safe_recv_result(result_conn, proc)
                    if proc.is_alive():
                        _terminate_process(proc)
                    proc.join(timeout=1)
                    result_conn.close()

                    if recv_ok:
                        status, _, msg = recv_value
                        if status == "ok":
                            total_success += 1
                            pbar.set_postfix({"ok": total_success, "fail": total_fail, "retry": total_requeued, "last": f"{subfolder}/{json_path.name} OK"})
                        else:
                            total_fail += 1
                            all_failures.append({"file": str(json_path), "error": msg})
                            pbar.set_postfix({"ok": total_success, "fail": total_fail, "retry": total_requeued, "last": f"{subfolder}/{json_path.name} FAIL"})
                    else:
                        total_fail += 1
                        all_failures.append({"file": str(json_path), "error": recv_value})
                        pbar.set_postfix({"ok": total_success, "fail": total_fail, "retry": total_requeued, "last": f"{subfolder}/{json_path.name} FAIL"})
                    pbar.update(1)
                elif not proc.is_alive():
                    proc.join(timeout=1)
                    result_conn.close()

                    total_fail += 1
                    all_failures.append({"file": str(json_path), "error": _worker_exit_error(proc)})
                    pbar.set_postfix({"ok": total_success, "fail": total_fail, "retry": total_requeued, "last": f"{subfolder}/{json_path.name} FAIL"})
                    pbar.update(1)
                else:
                    # Not done yet, keep in active list
                    still_active.append((json_path, output_dir, proc, result_conn, start_time))

            active = still_active

            # Small sleep to prevent busy waiting
            if active:
                time.sleep(0.1)
        elif pending and start_failed_this_round:
            time.sleep(0.5)

    pbar.close()

    print()
    print("=" * 60)
    print(f"[batch] TOTAL: {total_success} success, {total_fail} failed")
    print(f"[batch] Requeued on start failure: {total_requeued}")
    if total_success + total_fail > 0:
        print(f"[batch] Success rate: {total_success / (total_success + total_fail) * 100:.2f}%")

    if all_failures:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure_log = args.output_root / "failures.json"
        with open(failure_log, "w", encoding="utf-8") as f:
            json.dump(all_failures, f, indent=2, ensure_ascii=False)
        print(f"[batch] Failure details written to {failure_log}")


if __name__ == "__main__":
    main()
