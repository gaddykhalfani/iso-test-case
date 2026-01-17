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
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="ISO Dashboard Backend",
    description="API for running distillation column optimization algorithms",
    version="1.0.0"
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
# }

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
    except Exception as e:
        print(f"[READER] Error: {e}", flush=True)
        jobs[job_id]["status"] = "failed"
        _enqueue(loop, q, f"*** READER THREAD ERROR: {e}")

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
        "seed": 42           # PSO/GA only
      }

    If demo=True, runs demo_runner.py which simulates output (no Aspen required).
    """
    import time as time_module

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
    }

    if demo:
        # Run the demo runner which prints simulated progress and writes a demo JSON
        cmd = [sys.executable, os.path.join(REPO_ROOT, "demo_runner.py"), job_id, algorithm]
    else:
        # Run the appropriate optimizer as a subprocess
        script = ALGORITHM_SCRIPTS[algorithm]
        cmd = [sys.executable, os.path.join(REPO_ROOT, script), case]

        # Add algorithm-specific options
        if algorithm == "ISO":
            if no_sweep:
                cmd.append("--no-sweep")
        elif algorithm in ("PSO", "GA"):
            if n_particles:
                cmd.extend(["--n-particles", str(n_particles)])
            if n_iterations:
                cmd.extend(["--n-iterations", str(n_iterations)])
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
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"]),
        "total_jobs": len(jobs),
        "results_count": len(glob.glob(os.path.join(RESULTS_DIR, "*.json"))),
    }