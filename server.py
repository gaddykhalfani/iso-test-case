"""
server.py
FastAPI backend to start optimizer runs and stream live logs over WebSocket.

Usage:
  pip install fastapi uvicorn python-multipart
  uvicorn server:app --host 0.0.0.0 --port 8000

Supported Algorithms:
  - ISO: Iterative Sequential Optimization (main_sequential_optimizer.py)
  - PSO: Particle Swarm Optimization (pso_optimizer.py)
  - GA: Genetic Algorithm (ga_optimizer.py)

Notes:
- For real Aspen runs the subprocess should run on the same Windows machine
  that has Aspen + pywin32.
- For testing use demo mode (send {"demo":true}) which runs demo_runner.py.
"""
import asyncio
import subprocess
import sys
import os
import uuid
import threading
import glob
import json
import zipfile
from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# Optional Discord notifications
try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import Aspen file manager for concurrent run isolation
from aspen_file_manager import AspenFileManager, get_file_manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: cleanup orphaned temp files from previous crashed runs
    try:
        file_manager = get_file_manager()
        cleaned = file_manager.cleanup_old_copies(max_age_hours=24)
        if cleaned > 0:
            print(f"[STARTUP] Cleaned up {cleaned} orphaned temp files", flush=True)
    except Exception as e:
        print(f"[STARTUP] Warning: Failed to cleanup orphaned files: {e}", flush=True)

    yield  # Server runs here

    # Shutdown: could add cleanup here if needed
    print("[SHUTDOWN] Server shutting down", flush=True)


app = FastAPI(
    title="ISO Dashboard Backend",
    description="API for running distillation column optimization algorithms",
    version="1.0.0",
    lifespan=lifespan
)


# ════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ════════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    """Request model for starting an optimization run."""
    case: str
    algorithm: str = "ISO"  # ISO, PSO, GA
    demo: bool = False
    no_sweep: bool = False
    # PSO/GA specific options
    n_particles: Optional[int] = None
    n_iterations: Optional[int] = None
    seed: Optional[int] = None
    # Per-run config overrides (multi-run safe)
    config_overrides: Optional[Dict] = None


class JobStatus(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    case: str
    algorithm: str
    returncode: Optional[int] = None
    result_file: Optional[str] = None
    elapsed_seconds: Optional[float] = None

# Allow CORS for local dev (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Jobs store
jobs: Dict[str, Dict] = {}
# job structure:
# {
#   "process": subprocess.Popen | None,
#   "queue": asyncio.Queue,
#   "status": "starting"|"running"|"done"|"failed"|"killed",
#   "returncode": int | None,
#   "result_file": str | None,
#   "case": str,
#   "algorithm": str,
#   "start_time": float,
#   "convergence_history": List[Dict],  # Live convergence data for charts
#   "last_progress": Dict,  # Most recent progress update
#   "logs": List[str],  # Full log lines for detailed log panel
# }

# Discord webhook URL (can be set via API or environment)
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def send_discord_notification(job_result: dict, webhook_url: str = None):
    """Send a Discord notification when a job completes."""
    if not HAS_REQUESTS:
        print("[DISCORD] requests library not available, skipping notification")
        return False

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return False

    # Determine color based on status
    status = job_result.get("status", "done")
    color_map = {
        "done": 0x4fc3f7,    # Electric blue
        "failed": 0xef5350,  # Red
        "killed": 0xffb74d,  # Amber
    }
    color = color_map.get(status, 0x8b949e)

    # Build embed
    embed = {
        "title": f"{'✅' if status == 'done' else '❌'} Optimization {status.upper()}: {job_result.get('case', 'Unknown')}",
        "color": color,
        "fields": [
            {"name": "Algorithm", "value": job_result.get("algorithm", "ISO"), "inline": True},
            {"name": "Status", "value": status.upper(), "inline": True},
        ],
        "timestamp": datetime.now().astimezone().isoformat(),
        "footer": {"text": "Column Optimization Dashboard"}
    }

    # Add TAC if available
    best_tac = job_result.get("best_tac")
    if best_tac and best_tac < 1e10:
        embed["fields"].append({"name": "Best TAC", "value": f"${best_tac:,.0f}/year", "inline": True})

    # Add elapsed time
    elapsed = job_result.get("elapsed_seconds")
    if elapsed:
        embed["fields"].append({"name": "Duration", "value": f"{elapsed:.1f}s", "inline": True})

    # Add optimal config if available
    optimal = job_result.get("optimal", {})
    if optimal:
        config_str = f"NT={optimal.get('nt', '?')}, Feed={optimal.get('feed', '?')}, P={optimal.get('pressure', '?'):.3f} bar"
        embed["fields"].append({"name": "Optimal Config", "value": config_str, "inline": False})

    try:
        response = http_requests.post(url, json={"embeds": [embed]}, timeout=5)
        return response.status_code in (200, 204)
    except Exception as e:
        print(f"[DISCORD] Failed to send notification: {e}")
        return False

# Algorithm to script mapping
ALGORITHM_SCRIPTS = {
    "ISO": "main_sequential_optimizer.py",
    "PSO": "pso_optimizer.py",
    "GA": "ga_optimizer.py",
}

def _enqueue(loop, q: asyncio.Queue, item: str):
    """Schedule putting item into asyncio.Queue from any thread."""
    asyncio.run_coroutine_threadsafe(q.put(item), loop)

def _parse_progress(line: str) -> Optional[Dict]:
    """Parse a [PROGRESS] JSON line and return the data."""
    if "[PROGRESS]" in line:
        try:
            json_str = line.split("[PROGRESS]")[1].strip()
            return json.loads(json_str)
        except:
            pass
    return None

def _reader_thread(proc: subprocess.Popen, job_id: str, loop: asyncio.AbstractEventLoop):
    """Read lines from proc.stdout and push into job queue."""
    import time as time_module

    print(f"[READER] Started reader thread for job {job_id}", flush=True)
    q = jobs[job_id]["queue"]
    line_count = 0
    try:
        for raw in iter(proc.stdout.readline, ""):
            if raw is None:
                break
            line = raw.rstrip("\n")
            line_count += 1
            print(f"[READER] Line {line_count}: {line[:100]}", flush=True)  # Print first 100 chars
            _enqueue(loop, q, line)

            # Store log line for detailed log panel
            if "logs" not in jobs[job_id]:
                jobs[job_id]["logs"] = []
            jobs[job_id]["logs"].append(line)

            # Parse [PROGRESS] lines for live convergence tracking
            progress = _parse_progress(line)
            if progress:
                # Store last progress
                jobs[job_id]["last_progress"] = progress

                # Add to convergence history if it has best_tac
                if "best_tac" in progress and progress["best_tac"] is not None:
                    convergence_point = {
                        "iteration": progress.get("iteration", progress.get("current", 0)),
                        "best_tac": progress["best_tac"],
                        "phase": progress.get("phase", ""),
                        "algorithm": progress.get("algorithm", jobs[job_id].get("algorithm", ""))
                    }
                    jobs[job_id]["convergence_history"].append(convergence_point)

            # Try to capture final result file path if printed by the optimizer
            if "Results saved to:" in line:
                # Expect path after colon
                try:
                    path = line.split("Results saved to:")[-1].strip()
                    # If relative, make absolute relative to REPO_ROOT
                    if not os.path.isabs(path):
                        path = os.path.join(REPO_ROOT, path)
                    jobs[job_id]["result_file"] = path
                    _enqueue(loop, q, f"*** recorded result file: {path}")
                except Exception:
                    pass
        proc.stdout.close()
        proc.wait()
        jobs[job_id]["returncode"] = proc.returncode
        jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
        print(f"[READER] Process exited with code: {proc.returncode}", flush=True)
        _enqueue(loop, q, f"*** PROCESS EXIT: code={proc.returncode}")

        # Send Discord notification on job completion
        elapsed = None
        if jobs[job_id].get("start_time"):
            elapsed = round(time_module.time() - jobs[job_id]["start_time"], 1)

        # Get best TAC from last progress or convergence history
        best_tac = jobs[job_id].get("last_progress", {}).get("best_tac")
        if not best_tac and jobs[job_id].get("convergence_history"):
            best_tac = jobs[job_id]["convergence_history"][-1].get("best_tac")

        # Try to load optimal config from result file
        optimal = {}
        result_file = jobs[job_id].get("result_file")
        if result_file and os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    optimal = result_data.get("optimal", {})
                    if not best_tac:
                        best_tac = optimal.get("tac")
            except Exception:
                pass

        notification_data = {
            "job_id": job_id,
            "case": jobs[job_id].get("case"),
            "algorithm": jobs[job_id].get("algorithm", "ISO"),
            "status": jobs[job_id]["status"],
            "best_tac": best_tac,
            "elapsed_seconds": elapsed,
            "optimal": optimal,
        }

        # Check for webhook URL in job or global
        webhook_url = jobs[job_id].get("discord_webhook") or DISCORD_WEBHOOK_URL
        if webhook_url:
            send_discord_notification(notification_data, webhook_url)

        # Cleanup isolated Aspen file copy
        isolated_file = jobs[job_id].get("isolated_file")
        if isolated_file:
            try:
                file_manager = get_file_manager()
                file_manager.cleanup_run_copy(job_id)
                print(f"[READER] Cleaned up isolated file for job {job_id}", flush=True)
            except Exception as cleanup_error:
                print(f"[READER] Warning: Failed to cleanup isolated file: {cleanup_error}", flush=True)

    except Exception as e:
        print(f"[READER] Error: {e}", flush=True)
        jobs[job_id]["status"] = "failed"
        _enqueue(loop, q, f"*** READER THREAD ERROR: {e}")

# Maximum concurrent runs constant (used in /run endpoint)
MAX_CONCURRENT_RUNS_LIMIT = 4


@app.post("/run")
async def start_run(payload: Dict):
    """
    Start an optimizer run.

    JSON payload:
      {
        "case": "Case1_COL2",
        "algorithm": "ISO",  # ISO, PSO, or GA
        "demo": false,
        "no_sweep": false,
        "n_particles": 20,   # PSO/GA only
        "n_iterations": 50,  # PSO/GA only
        "seed": 42,          # PSO/GA only
        "config_overrides": {  # Per-run config (multi-run safe)
          "nt_bounds": [15, 100],
          "feed_bounds": [5, 80],
          "pressure_bounds": [0.1, 1.5],
          "initial_nt": 30,
          "initial_feed": 20,
          "initial_pressure": 0.3
        }
      }

    If demo=True, runs demo_runner.py which simulates output (no Aspen required).

    Config overrides are saved to a run-specific JSON file to ensure
    multi-run safety (concurrent runs don't interfere with each other).

    Maximum concurrent runs: 4 (to prevent resource exhaustion)
    """
    import time as time_module

    # Check concurrent run limit
    active_count = len([j for j in jobs.values() if j["status"] == "running"])
    if active_count >= MAX_CONCURRENT_RUNS_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent runs ({MAX_CONCURRENT_RUNS_LIMIT}) reached. Please wait for a run to complete."
        )

    case = payload.get("case")
    if not case:
        raise HTTPException(status_code=400, detail="case is required")

    algorithm = payload.get("algorithm", "ISO").upper()
    if algorithm not in ALGORITHM_SCRIPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm. Supported: {list(ALGORITHM_SCRIPTS.keys())}"
        )

    no_sweep = payload.get("no_sweep", False)
    demo = payload.get("demo", False)

    # PSO/GA specific options
    n_particles = payload.get("n_particles")
    n_iterations = payload.get("n_iterations")
    seed = payload.get("seed")

    # Per-run config overrides (multi-run safe)
    config_overrides = payload.get("config_overrides")

    job_id = str(uuid.uuid4())
    q = asyncio.Queue()
    jobs[job_id] = {
        "process": None,
        "queue": q,
        "status": "starting",
        "returncode": None,
        "result_file": None,
        "case": case,
        "algorithm": algorithm,
        "start_time": time_module.time(),
        "convergence_history": [],  # Live convergence data for charts
        "last_progress": {},  # Most recent progress update
        "config_overrides": config_overrides,  # Track what config was used
        "logs": [],  # Full log lines for detailed log panel
        "discord_webhook": payload.get("discord_webhook"),  # Per-run webhook URL
        "isolated_file": None,  # Path to isolated Aspen file copy
    }

    # Create isolated Aspen file copy for this run (prevents concurrent access conflicts)
    isolated_file_path = None
    if not demo:
        try:
            file_manager = get_file_manager()
            isolated_file_path = file_manager.create_run_copy(case, job_id)
            jobs[job_id]["isolated_file"] = isolated_file_path
            print(f"[SERVER] Created isolated Aspen file: {isolated_file_path}", flush=True)
        except Exception as e:
            print(f"[SERVER] Warning: Failed to create isolated file copy: {e}", flush=True)
            print(f"[SERVER] Proceeding with shared master file (may cause conflicts)", flush=True)

    # Create run-specific config file if overrides provided
    config_file_path = None
    if config_overrides and not demo:
        try:
            # Import config module functions
            from config import create_run_config, save_run_config

            # Create run-specific config (isolated copy with overrides)
            run_config = create_run_config(case, config_overrides)
            if run_config:
                # Save to a run-specific JSON file
                config_file_path = save_run_config(run_config, job_id, RESULTS_DIR)
                print(f"[SERVER] Created run-specific config: {config_file_path}", flush=True)
                jobs[job_id]["config_file"] = config_file_path
        except Exception as e:
            print(f"[SERVER] Warning: Failed to create run config: {e}", flush=True)
            # Continue without config file - optimizer will use defaults

    if demo:
        # Run the demo runner which prints simulated progress and writes a demo JSON
        cmd = [sys.executable, os.path.join(REPO_ROOT, "demo_runner.py"), job_id, algorithm]
    else:
        # Run the appropriate optimizer as a subprocess
        script = ALGORITHM_SCRIPTS[algorithm]
        cmd = [sys.executable, os.path.join(REPO_ROOT, script), case]

        # Add config file argument if we have run-specific config
        if config_file_path:
            cmd.extend(["--config-file", config_file_path])

        # Add isolated file argument if we have an isolated copy
        if isolated_file_path:
            cmd.extend(["--isolated-file", isolated_file_path])

        # Add algorithm-specific options
        if algorithm == "ISO":
            if no_sweep:
                cmd.append("--no-sweep")
        elif algorithm == "PSO":
            if n_particles:
                cmd.extend(["--n-particles", str(n_particles)])
            if n_iterations:
                cmd.extend(["--n-iterations", str(n_iterations)])
            if seed:
                cmd.extend(["--seed", str(seed)])
            if demo:
                cmd.append("--demo")
        elif algorithm == "GA":
            if n_particles:  # Used as pop_size for GA
                cmd.extend(["--pop-size", str(n_particles)])
            if n_iterations:  # Used as n_generations for GA
                cmd.extend(["--n-generations", str(n_iterations)])
            if seed:
                cmd.extend(["--seed", str(seed)])
            if demo:
                cmd.append("--demo")

    # Start process
    # On Windows, we need to pass environment and use CREATE_NEW_CONSOLE for COM
    import platform
    popen_kwargs = {
        'cwd': REPO_ROOT,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'bufsize': 1,
        'universal_newlines': True,
        'env': os.environ.copy(),  # Pass environment variables
    }

    # On Windows, allow COM to work properly
    if platform.system() == 'Windows':
        # CREATE_NEW_PROCESS_GROUP allows the process to run independently
        popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

    # Debug: Print the command being executed
    print(f"[SERVER] Starting subprocess: {' '.join(cmd)}", flush=True)
    print(f"[SERVER] Working directory: {REPO_ROOT}", flush=True)

    proc = subprocess.Popen(cmd, **popen_kwargs)
    jobs[job_id]["process"] = proc
    jobs[job_id]["status"] = "running"

    print(f"[SERVER] Process started with PID: {proc.pid}", flush=True)

    # Start reader thread to push stdout -> asyncio.Queue
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=_reader_thread, args=(proc, job_id, loop), daemon=True)
    t.start()

    return {"job_id": job_id, "ws": f"/ws/{job_id}"}

@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Get status of a specific job."""
    import time as time_module

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    elapsed = None
    if job.get("start_time"):
        elapsed = round(time_module.time() - job["start_time"], 1)

    return {
        "job_id": job_id,
        "status": job["status"],
        "returncode": job["returncode"],
        "case": job.get("case"),
        "algorithm": job.get("algorithm", "ISO"),
        "result_file": job.get("result_file"),
        "elapsed_seconds": elapsed,
        "last_progress": job.get("last_progress", {}),
    }

@app.get("/convergence/all")
async def get_all_convergence():
    """
    Get convergence history for all active/recent jobs.
    Used for the combined convergence chart overlay.
    NOTE: This route MUST be defined before /convergence/{job_id} to match correctly.
    """
    import time as time_module

    all_convergence = []
    for jid, job in jobs.items():
        # Include running jobs and recently completed ones
        if job["status"] in ("running", "done", "failed", "killed"):
            elapsed = None
            if job.get("start_time"):
                elapsed = round(time_module.time() - job["start_time"], 1)

            all_convergence.append({
                "job_id": jid,
                "case": job.get("case"),
                "algorithm": job.get("algorithm", "ISO"),
                "status": job["status"],
                "elapsed_seconds": elapsed,
                "convergence_history": job.get("convergence_history", []),
                "last_progress": job.get("last_progress", {}),
            })

    return {
        "jobs": all_convergence,
        "count": len(all_convergence),
    }


@app.get("/convergence/{job_id}")
async def get_convergence(job_id: str):
    """Get convergence history for a job (for live charts)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job_id,
        "algorithm": job.get("algorithm", "ISO"),
        "status": job["status"],
        "convergence_history": job.get("convergence_history", []),
        "last_progress": job.get("last_progress", {}),
    }

@app.post("/kill/{job_id}")
async def kill_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    proc = job.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        job["status"] = "killed"

        # Cleanup isolated Aspen file copy
        isolated_file = job.get("isolated_file")
        if isolated_file:
            try:
                file_manager = get_file_manager()
                file_manager.cleanup_run_copy(job_id)
                print(f"[SERVER] Cleaned up isolated file for killed job {job_id}", flush=True)
            except Exception as cleanup_error:
                print(f"[SERVER] Warning: Failed to cleanup isolated file: {cleanup_error}", flush=True)

        return {"killed": True}
    return {"killed": False, "reason": "no running process"}

@app.get("/jobs")
async def list_jobs():
    return [{"job_id": jid, "status": jobs[jid]["status"], "case": jobs[jid]["case"]} for jid in jobs]

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    if job_id not in jobs:
        await websocket.close(code=1000)
        return
    await websocket.accept()
    job = jobs[job_id]
    q: asyncio.Queue = job["queue"]
    try:
        await websocket.send_text(f"*** connected to job {job_id} (case={job.get('case')})")
        # Stream queue -> websocket until process done and queue empty
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if job["status"] in ("done", "failed", "killed") and q.empty():
                    await websocket.send_text("*** no more data. closing")
                    await websocket.close()
                    break
                continue

            await websocket.send_text(line)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(f"*** server error: {e}")
        except:
            pass
        await websocket.close()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/results")
async def list_results():
    """List all result JSON files in the results directory."""
    results = []
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))

    for filepath in sorted(json_files, key=os.path.getmtime, reverse=True):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract summary info
            optimal = data.get('optimal', {})
            metadata = data.get('metadata', {})
            stats = data.get('statistics', {})

            results.append({
                "filename": filename,
                "filepath": filepath,
                "case_name": metadata.get('case_name', data.get('case_name', 'Unknown')),
                "algorithm": data.get('algorithm', metadata.get('algorithm', 'ISO')),
                "optimal_tac": optimal.get('tac'),
                "optimal_nt": optimal.get('nt'),
                "optimal_feed": optimal.get('feed'),
                "optimal_pressure": optimal.get('pressure'),
                "total_evaluations": stats.get('total_evaluations'),
                "time_seconds": stats.get('time_seconds'),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "filepath": filepath,
                "error": str(e),
            })

    return {"results": results, "count": len(results)}


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Get a specific result JSON file."""
    # Sanitize filename to prevent directory traversal
    filename = os.path.basename(filename)
    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Result file not found")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {e}")


@app.get("/results/{filename}/download")
async def download_result(filename: str):
    """Download a result file."""
    filename = os.path.basename(filename)
    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        filepath,
        media_type="application/json",
        filename=filename
    )


@app.get("/plots")
async def list_plots():
    """List all plot PNG files in the results directory."""
    plots = []
    png_files = glob.glob(os.path.join(RESULTS_DIR, "*.png"))

    for filepath in sorted(png_files, key=os.path.getmtime, reverse=True):
        filename = os.path.basename(filepath)
        plots.append({
            "filename": filename,
            "filepath": filepath,
            "size_bytes": os.path.getsize(filepath),
            "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
        })

    return {"plots": plots, "count": len(plots)}


@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """Get a specific plot PNG file."""
    filename = os.path.basename(filename)
    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Plot file not found")

    return FileResponse(filepath, media_type="image/png")


# ════════════════════════════════════════════════════════════════════════════
# LOGS ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@app.get("/logs/{job_id}")
async def get_logs(job_id: str, tail: int = 500, filter_text: str = None):
    """
    Get logs for a specific job.

    Args:
        job_id: The job ID
        tail: Number of recent lines to return (default 500)
        filter_text: Optional text to filter log lines
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = job.get("logs", [])

    # Apply filter if provided
    if filter_text:
        logs = [line for line in logs if filter_text.lower() in line.lower()]

    # Return tail lines
    if tail > 0 and len(logs) > tail:
        logs = logs[-tail:]

    return {
        "job_id": job_id,
        "status": job["status"],
        "total_lines": len(job.get("logs", [])),
        "returned_lines": len(logs),
        "logs": logs,
    }


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@app.get("/stats")
async def get_statistics():
    """
    Get aggregated statistics from all historical results.
    Used for the Statistics Dashboard.
    """
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))

    stats = {
        "total_runs": 0,
        "by_algorithm": {"ISO": [], "PSO": [], "GA": []},
        "by_case": {},
        "best_tac_ever": None,
        "best_run": None,
        "total_evaluations": 0,
        "total_time_seconds": 0,
        "success_count": 0,
        "failure_count": 0,
    }

    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip run config files
            if "run_config" in os.path.basename(filepath):
                continue

            stats["total_runs"] += 1

            # Extract data
            optimal = data.get('optimal', {})
            metadata = data.get('metadata', {})
            statistics = data.get('statistics', {})
            convergence = data.get('convergence', {})

            algorithm = data.get('algorithm', metadata.get('algorithm', 'ISO'))
            case_name = metadata.get('case_name', data.get('case_name', 'Unknown'))
            tac = optimal.get('tac')
            time_seconds = statistics.get('time_seconds', 0)
            total_evals = statistics.get('total_evaluations', 0)
            converged = convergence.get('converged', True)

            # Track by algorithm
            if algorithm in stats["by_algorithm"]:
                stats["by_algorithm"][algorithm].append({
                    "filename": os.path.basename(filepath),
                    "case": case_name,
                    "tac": tac,
                    "time_seconds": time_seconds,
                    "evaluations": total_evals,
                    "converged": converged,
                    "optimal": optimal,
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
                })

            # Track by case
            if case_name not in stats["by_case"]:
                stats["by_case"][case_name] = []
            stats["by_case"][case_name].append({
                "algorithm": algorithm,
                "tac": tac,
                "time_seconds": time_seconds,
            })

            # Track best TAC
            if tac and (stats["best_tac_ever"] is None or tac < stats["best_tac_ever"]):
                stats["best_tac_ever"] = tac
                stats["best_run"] = {
                    "filename": os.path.basename(filepath),
                    "case": case_name,
                    "algorithm": algorithm,
                    "optimal": optimal,
                }

            # Aggregate totals
            stats["total_evaluations"] += total_evals
            stats["total_time_seconds"] += time_seconds
            if converged:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1

        except Exception as e:
            print(f"[STATS] Error reading {filepath}: {e}")
            continue

    # Calculate averages
    for algo in ["ISO", "PSO", "GA"]:
        runs = stats["by_algorithm"][algo]
        if runs:
            valid_tacs = [r["tac"] for r in runs if r["tac"] and r["tac"] < 1e10]
            valid_times = [r["time_seconds"] for r in runs if r["time_seconds"]]
            stats["by_algorithm"][algo] = {
                "runs": runs,
                "count": len(runs),
                "avg_tac": sum(valid_tacs) / len(valid_tacs) if valid_tacs else None,
                "min_tac": min(valid_tacs) if valid_tacs else None,
                "max_tac": max(valid_tacs) if valid_tacs else None,
                "avg_time": sum(valid_times) / len(valid_times) if valid_times else None,
            }
        else:
            stats["by_algorithm"][algo] = {"runs": [], "count": 0}

    return stats


# ════════════════════════════════════════════════════════════════════════════
# EXPORT ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/export/results/csv")
async def export_results_csv():
    """Export all results as CSV."""
    import csv
    from io import StringIO

    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Filename", "Case", "Algorithm", "TAC", "NT", "Feed", "Pressure",
        "T_reb", "Evaluations", "Time (s)", "Converged", "Modified"
    ])

    for filepath in sorted(json_files, key=os.path.getmtime, reverse=True):
        try:
            # Skip config files
            if "run_config" in os.path.basename(filepath):
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            optimal = data.get('optimal', {})
            metadata = data.get('metadata', {})
            statistics = data.get('statistics', {})
            convergence = data.get('convergence', {})

            writer.writerow([
                os.path.basename(filepath),
                metadata.get('case_name', data.get('case_name', '')),
                data.get('algorithm', metadata.get('algorithm', 'ISO')),
                optimal.get('tac', ''),
                optimal.get('nt', ''),
                optimal.get('feed', ''),
                optimal.get('pressure', ''),
                optimal.get('T_reb', ''),
                statistics.get('total_evaluations', ''),
                statistics.get('time_seconds', ''),
                convergence.get('converged', ''),
                datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            ])
        except Exception:
            continue

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=optimization_results.csv"}
    )


@app.get("/export/plots/zip")
async def export_plots_zip():
    """Export all plots as a ZIP file."""
    png_files = glob.glob(os.path.join(RESULTS_DIR, "*.png"))

    if not png_files:
        raise HTTPException(status_code=404, detail="No plot files found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for png_path in png_files:
            zf.write(png_path, os.path.basename(png_path))

    zip_buffer.seek(0)
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"}
    )


# ════════════════════════════════════════════════════════════════════════════
# DISCORD WEBHOOK CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@app.post("/discord/webhook")
async def set_discord_webhook(payload: Dict):
    """
    Set the global Discord webhook URL.

    JSON payload:
      {"webhook_url": "https://discord.com/api/webhooks/..."}
    """
    global DISCORD_WEBHOOK_URL

    webhook_url = payload.get("webhook_url", "")
    DISCORD_WEBHOOK_URL = webhook_url

    return {"status": "ok", "webhook_set": bool(webhook_url)}


@app.post("/discord/test")
async def test_discord_webhook(payload: Dict):
    """
    Test the Discord webhook by sending a test message.

    JSON payload:
      {"webhook_url": "https://discord.com/api/webhooks/..."}  (optional, uses global if not provided)
    """
    webhook_url = payload.get("webhook_url") or DISCORD_WEBHOOK_URL

    if not webhook_url:
        raise HTTPException(status_code=400, detail="No webhook URL configured")

    test_data = {
        "case": "Test Case",
        "algorithm": "TEST",
        "status": "done",
        "best_tac": 123456.78,
        "elapsed_seconds": 42.0,
        "optimal": {"nt": 30, "feed": 15, "pressure": 0.3},
    }

    success = send_discord_notification(test_data, webhook_url)
    return {"success": success, "message": "Test notification sent" if success else "Failed to send notification"}


# ════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════

# Maximum concurrent runs allowed
MAX_CONCURRENT_RUNS = 4


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    active_jobs = [j for j in jobs.values() if j["status"] == "running"]
    return {
        "status": "healthy",
        "active_jobs": len(active_jobs),
        "max_concurrent": MAX_CONCURRENT_RUNS,
        "total_jobs": len(jobs),
        "results_count": len(glob.glob(os.path.join(RESULTS_DIR, "*.json"))),
    }


@app.get("/active-runs")
async def get_active_runs():
    """Get list of currently active/running jobs."""
    import time as time_module

    active = []
    for jid, job in jobs.items():
        if job["status"] == "running":
            elapsed = None
            if job.get("start_time"):
                elapsed = round(time_module.time() - job["start_time"], 1)

            active.append({
                "job_id": jid,
                "case": job.get("case"),
                "algorithm": job.get("algorithm", "ISO"),
                "status": job["status"],
                "elapsed_seconds": elapsed,
                "last_progress": job.get("last_progress", {}),
                "config_overrides": job.get("config_overrides"),
            })

    return {
        "active_runs": active,
        "count": len(active),
        "max_concurrent": MAX_CONCURRENT_RUNS,
    }