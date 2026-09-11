"""
Data Scientist Agent 2.0 - FastAPI application.

Run it with:  uvicorn main:app --reload

The routes here are thin on purpose. Each one validates its input, calls into
the agent package or the database, and renders a template. All of the actual
data-science work lives in agent/pipeline.py.
"""

import hashlib
import os
import secrets
import shutil
import time
import traceback
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import db
from agent import models as model_library
from agent import pipeline, profiling, report

app = FastAPI(title="Data Scientist Agent 2.0")

# Vercel's deployed filesystem is read-only except for /tmp.
# Keep normal local development paths unchanged, but use writable temporary
# directories on Vercel for uploaded datasets and generated charts.
if os.getenv("VERCEL"):
    RUNTIME_DIR = "/tmp/datanexus"
    RUNTIME_UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads")
    RUNTIME_CHART_DIR = os.path.join(RUNTIME_DIR, "charts")
    os.makedirs(RUNTIME_UPLOAD_DIR, exist_ok=True)
    os.makedirs(RUNTIME_CHART_DIR, exist_ok=True)
else:
    RUNTIME_UPLOAD_DIR = config.UPLOAD_DIR
    RUNTIME_CHART_DIR = config.CHART_DIR

# Mount the runtime chart directory before the general /static mount so the
# existing /static/charts/<filename> URLs continue to work on Vercel.
app.mount("/static/charts", StaticFiles(directory=RUNTIME_CHART_DIR), name="charts")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)


def _materialize_dataset(dataset):
    """Return a readable local CSV path, restoring it from DB when needed.

    Vercel instances have ephemeral /tmp storage, so a later request may land
    on a different instance. The uploaded CSV is therefore restored from the
    persistent MySQL blob when the original runtime path is unavailable.
    """
    stored_path = dataset.get("stored_path")
    if stored_path and os.path.exists(stored_path):
        return stored_path

    file_content = dataset.get("file_content")
    if file_content is None:
        raise FileNotFoundError("Uploaded CSV is no longer available on this server instance.")

    extension = os.path.splitext(dataset.get("original_filename") or "dataset.csv")[1].lower() or ".csv"
    restored_path = os.path.join(RUNTIME_UPLOAD_DIR, f"dataset_{dataset['id']}{extension}")
    with open(restored_path, "wb") as handle:
        handle.write(file_content)
    return restored_path


# --------------------------------------------------------------------------
# Template helpers
# --------------------------------------------------------------------------

def comma(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return value


def score(value, places=4):
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "n/a"


def percent(value, places=1):
    try:
        return f"{float(value):.{places}f}%"
    except (TypeError, ValueError):
        return "n/a"


templates.env.filters["comma"] = comma
templates.env.filters["score"] = score
templates.env.filters["percent"] = percent


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    ok, message = db.check_connection()
    print(f"[startup] {message}")
    if ok:
        try:
            db.init_database()
            print("[startup] Database and tables are ready.")
        except Exception as exc:                 # noqa: BLE001
            print(f"[startup] Could not create the schema: {exc}")
    else:
        print("[startup] Fix the MySQL settings in your .env file before uploading a dataset.")
    if config.GEMINI_ENABLED:
        print(f"[startup] Gemini enabled using model {config.GEMINI_MODEL}.")
    else:
        print("[startup] No GEMINI_API_KEY set - the built-in rule-based analyst will be used.")


# --------------------------------------------------------------------------
# Pages that need no analysis
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "gemini_enabled": config.GEMINI_ENABLED,
        "model_count": len([m for m in model_library.MODEL_LIBRARY.values() if not m["is_baseline"]]),
    })


@app.get("/workspace", response_class=HTMLResponse)
def workspace(request: Request, error: str = None):
    return templates.TemplateResponse("workspace.html", {
        "request": request,
        "error": error,
        "gemini_enabled": config.GEMINI_ENABLED,
        "max_mb": config.MAX_UPLOAD_MB,
        "recent": _safe_recent(),
    })


@app.get("/models", response_class=HTMLResponse)
def model_catalogue(request: Request):
    """Shows exactly which models the agent is allowed to run, and nothing else."""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "classification": model_library.list_library("classification"),
        "regression": model_library.list_library("regression"),
    })


def _safe_recent():
    """The recent-analyses list must never be the reason a page fails to load."""
    try:
        return db.list_recent_analyses(6)
    except Exception:                            # noqa: BLE001
        return []


# --------------------------------------------------------------------------
# Upload and analysis
# --------------------------------------------------------------------------

@app.post("/upload")
def upload(background_tasks: BackgroundTasks,
           dataset: UploadFile = File(...),
           objective: str = Form(...)):
    """Validate the file, store it, create the analysis and start the agent."""
    objective = (objective or "").strip()
    if len(objective) < 10:
        return RedirectResponse(
            "/workspace?error=Please describe your objective in a full sentence "
            "so the agent knows what to look for.", status_code=303)

    extension = os.path.splitext(dataset.filename or "")[1].lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        return RedirectResponse(
            f"/workspace?error=Only CSV files are supported at the moment. "
            f"You uploaded '{extension or 'a file with no extension'}'.", status_code=303)

    safe_name = f"{int(time.time())}_{secrets.token_hex(4)}{extension}"
    stored_path = os.path.join(RUNTIME_UPLOAD_DIR, safe_name)

    with open(stored_path, "wb") as handle:
        shutil.copyfileobj(dataset.file, handle)

    file_size = os.path.getsize(stored_path)
    if file_size > config.MAX_UPLOAD_BYTES:
        os.remove(stored_path)
        return RedirectResponse(
            f"/workspace?error=That file is {file_size / 1024 / 1024:.1f} MB, over the "
            f"{config.MAX_UPLOAD_MB} MB limit.", status_code=303)
    if file_size == 0:
        os.remove(stored_path)
        return RedirectResponse("/workspace?error=That file is empty.", status_code=303)

    try:
        frame = pipeline.load_csv(stored_path)
    except Exception as exc:                     # noqa: BLE001
        os.remove(stored_path)
        return RedirectResponse(f"/workspace?error={exc}", status_code=303)

    if frame.shape[0] < 10:
        os.remove(stored_path)
        return RedirectResponse(
            f"/workspace?error=This file has only {frame.shape[0]} row(s). At least 10 are "
            f"needed for any meaningful analysis.", status_code=303)

    try:
        with open(stored_path, "rb") as handle:
            file_content = handle.read()
        quick_profile = profiling.profile_dataframe(frame, sample_rows=5)
        dataset_id = db.create_dataset(dataset.filename, stored_path, file_size,
                                       frame.shape[0], frame.shape[1], quick_profile,
                                       file_content)
        analysis_id = db.create_analysis(dataset_id, objective)
        db.create_steps(analysis_id, pipeline.STEPS)
    except Exception as exc:                     # noqa: BLE001
        return RedirectResponse(
            f"/workspace?error=Could not save to MySQL: {exc}. Check the connection settings "
            f"in your .env file.", status_code=303)

    background_tasks.add_task(run_analysis_task, analysis_id, None)
    return RedirectResponse(f"/analysis/{analysis_id}", status_code=303)


def run_analysis_task(analysis_id, forced_target):
    """
    The background job.

    Progress is written to MySQL as each step really starts and finishes, so the
    progress bar on screen reflects the agent's actual position rather than a
    timer pretending to be one.
    """
    analysis = db.get_analysis(analysis_id)
    if analysis is None:
        return

    dataset = db.get_dataset(analysis["dataset_id"])
    progress = pipeline.Progress(
        on_start=lambda key: db.start_step(analysis_id, key),
        on_finish=lambda key, detail, status: db.finish_step(analysis_id, key, detail, status),
        on_update=lambda key, detail: db.run(
            "UPDATE analysis_steps SET detail = %s WHERE analysis_id = %s AND step_key = %s",
            (detail, analysis_id, key)),
    )

    db.update_analysis(analysis_id, status="running", error_message=None)

    try:
        stored_path = _materialize_dataset(dataset)
        frame = pipeline.load_csv(stored_path)
        results = pipeline.run_analysis(
            frame, analysis["objective"], progress=progress,
            forced_target=forced_target, analysis_id=analysis_id,
            chart_dir=RUNTIME_CHART_DIR)
    except Exception as exc:                     # noqa: BLE001
        traceback.print_exc()
        db.update_analysis(analysis_id, status="failed",
                           error_message=f"{type(exc).__name__}: {exc}"[:2000])
        return

    db.save_results(analysis_id, results)
    status = results["status"]

    if status == "needs_target":
        db.update_analysis(analysis_id, status="needs_target")
        return

    if status == "failed":
        db.update_analysis(analysis_id, status="failed", error_message=results.get("error"))
        return

    fields = {
        "status": "completed",
        "problem_type": results.get("problem_type"),
        "target_column": results.get("target_column"),
        "ai_source": results.get("ai_source"),
        "completed_at": datetime.now(),
    }
    if status == "completed":
        fields.update({
            "evaluation_metric": results["metric"],
            "best_model": results["best"]["model_name"],
            "best_score": results["best"]["primary_score"],
            "objective_status": results["objective_validation"]["status"],
            "objective_reason": results["objective_validation"]["reason"][:2000],
        })
        for result in results["model_results"]:
            result["is_best"] = (result["model_name"] == results["best"]["model_name"])
            db.save_model_result(analysis_id, result)

    db.update_analysis(analysis_id, **fields)
    db.save_report(analysis_id, report.build_markdown(results, dataset["original_filename"]))


# --------------------------------------------------------------------------
# Progress, results and report
# --------------------------------------------------------------------------

@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
def analysis_page(request: Request, analysis_id: int):
    analysis = _require_analysis(analysis_id)
    dataset = db.get_dataset(analysis["dataset_id"])

    if analysis["status"] == "completed":
        return RedirectResponse(f"/analysis/{analysis_id}/results", status_code=303)

    return templates.TemplateResponse("progress.html", {
        "request": request,
        "analysis": analysis,
        "dataset": dataset,
        "steps": db.get_steps(analysis_id),
        "results": analysis.get("results"),
    })


@app.get("/analysis/{analysis_id}/status")
def analysis_status(analysis_id: int):
    """Polled by the progress page. Returns only what the page needs to redraw."""
    analysis = db.get_analysis(analysis_id)
    if analysis is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    steps = db.get_steps(analysis_id)
    done = len([s for s in steps if s["status"] in ("done", "skipped")])
    return {
        "status": analysis["status"],
        "error": analysis["error_message"],
        "percent": int(100 * done / len(steps)) if steps else 0,
        "steps": [
            {"key": s["step_key"], "label": s["step_label"],
             "status": s["status"], "detail": s["detail"]}
            for s in steps
        ],
    }


@app.post("/analysis/{analysis_id}/target")
def confirm_target(background_tasks: BackgroundTasks, analysis_id: int,
                   target_column: str = Form(...)):
    """The user has told us which column to predict, so run the agent again."""
    analysis = _require_analysis(analysis_id)
    dataset = db.get_dataset(analysis["dataset_id"])

    stored_path = _materialize_dataset(dataset)
    frame = pipeline.load_csv(stored_path)
    if target_column not in frame.columns:
        raise HTTPException(status_code=400, detail="That column is not in the dataset.")

    db.run("UPDATE analysis_steps SET status = 'pending', detail = NULL, started_at = NULL, "
           "finished_at = NULL WHERE analysis_id = %s", (analysis_id,))
    db.update_analysis(analysis_id, status="queued", target_column=target_column)
    background_tasks.add_task(run_analysis_task, analysis_id, target_column)
    return RedirectResponse(f"/analysis/{analysis_id}", status_code=303)


@app.get("/analysis/{analysis_id}/results", response_class=HTMLResponse)
def results_page(request: Request, analysis_id: int):
    analysis = _require_analysis(analysis_id)
    if analysis["status"] in ("queued", "running", "needs_target"):
        return RedirectResponse(f"/analysis/{analysis_id}", status_code=303)

    results = analysis.get("results")
    if not results:
        return templates.TemplateResponse("progress.html", {
            "request": request,
            "analysis": analysis,
            "dataset": db.get_dataset(analysis["dataset_id"]),
            "steps": db.get_steps(analysis_id),
            "results": None,
        })

    return templates.TemplateResponse("results.html", {
        "request": request,
        "analysis": analysis,
        "dataset": db.get_dataset(analysis["dataset_id"]),
        "r": results,
    })


@app.get("/report/{analysis_id}", response_class=HTMLResponse)
def report_page(request: Request, analysis_id: int):
    analysis = _require_analysis(analysis_id)
    results = analysis.get("results")
    if not results:
        return RedirectResponse(f"/analysis/{analysis_id}", status_code=303)

    return templates.TemplateResponse("report.html", {
        "request": request,
        "analysis": analysis,
        "dataset": db.get_dataset(analysis["dataset_id"]),
        "r": results,
        "generated": datetime.now().strftime("%d %B %Y, %H:%M"),
    })


def _require_analysis(analysis_id):
    analysis = db.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="That analysis does not exist.")
    return analysis


# --------------------------------------------------------------------------
# Minimal accounts - only for downloading the report
# --------------------------------------------------------------------------
#
# This is deliberately the smallest thing that works: a salted PBKDF2 password
# hash in MySQL, and a one-use token held in memory that is good for a single
# download. There are no cookies, no sessions and no JWTs, because nothing else
# in the application is behind a login.

DOWNLOAD_TOKENS = {}          # token -> {"analysis_id": int, "email": str, "expires": float}
TOKEN_LIFETIME_SECONDS = 600


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${digest.hex()}"


def verify_password(password, stored):
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, analysis_id: int, error: str = None):
    _require_analysis(analysis_id)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "analysis_id": analysis_id,
        "error": error,
    })


@app.post("/login")
def login(analysis_id: int = Form(...), email: str = Form(...), password: str = Form(...)):
    """
    Sign in, or create the account on the spot if the email is new.

    Combining the two keeps the flow to a single screen, which is the whole
    point of only asking for an account at the download step.
    """
    email = (email or "").strip().lower()
    if "@" not in email or len(password) < 6:
        return RedirectResponse(
            f"/login?analysis_id={analysis_id}&error=Enter a valid email address and a "
            f"password of at least 6 characters.", status_code=303)

    try:
        user = db.get_user_by_email(email)
        if user is None:
            db.create_user(email, hash_password(password))
        elif not verify_password(password, user["password_hash"]):
            return RedirectResponse(
                f"/login?analysis_id={analysis_id}&error=That password does not match the "
                f"account already registered with this email.", status_code=303)
    except Exception as exc:                     # noqa: BLE001
        return RedirectResponse(
            f"/login?analysis_id={analysis_id}&error=Could not reach the database: {exc}",
            status_code=303)

    token = secrets.token_urlsafe(24)
    DOWNLOAD_TOKENS[token] = {
        "analysis_id": analysis_id,
        "email": email,
        "expires": time.time() + TOKEN_LIFETIME_SECONDS,
    }
    return RedirectResponse(f"/download-report/{analysis_id}?token={token}", status_code=303)


@app.get("/download-report/{analysis_id}")
def download_report(analysis_id: int, token: str = ""):
    entry = DOWNLOAD_TOKENS.get(token)
    if entry is None or entry["analysis_id"] != analysis_id or entry["expires"] < time.time():
        DOWNLOAD_TOKENS.pop(token, None)
        return RedirectResponse(
            f"/login?analysis_id={analysis_id}&error=That download link has expired. "
            f"Please sign in again.", status_code=303)

    DOWNLOAD_TOKENS.pop(token, None)             # one download per sign-in

    stored = db.get_report(analysis_id)
    if stored is None or not stored.get("report_md"):
        analysis = _require_analysis(analysis_id)
        results = analysis.get("results")
        if not results:
            raise HTTPException(status_code=404, detail="No report has been generated yet.")
        dataset = db.get_dataset(analysis["dataset_id"])
        markdown = report.build_markdown(results, dataset["original_filename"])
        db.save_report(analysis_id, markdown)
    else:
        markdown = stored["report_md"]

    filename = f"data_science_report_{analysis_id}.md"
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    return templates.TemplateResponse("history.html", {
        "request": request,
        "analyses": _safe_recent(),
    })
