"""
MySQL access layer.

Deliberately thin: a connection helper, three query helpers, and the small set
of read/write functions the application actually needs. No ORM, no repository
classes - just functions that take arguments and return dictionaries.
"""

import json
import os
import re

import mysql.connector

import config

SCHEMA_PATH = os.path.join(config.BASE_DIR, "schema.sql")


# --------------------------------------------------------------------------
# Connection helpers
# --------------------------------------------------------------------------

def get_connection(with_database=True):
    """Open a MySQL connection. Caller is responsible for closing it."""
    params = {
        "host": config.MYSQL_HOST,
        "port": config.MYSQL_PORT,
        "user": config.MYSQL_USER,
        "password": config.MYSQL_PASSWORD,
    }
    if with_database:
        params["database"] = config.MYSQL_DATABASE
    return mysql.connector.connect(**params)


def run(sql, params=None):
    """Execute a write statement and return the new row id (0 if not an insert)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()
        return new_id
    finally:
        conn.close()


def fetch_all(sql, params=None):
    """Execute a SELECT and return a list of dictionaries."""
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()


def fetch_one(sql, params=None):
    """Execute a SELECT and return the first row as a dictionary, or None."""
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


# --------------------------------------------------------------------------
# Schema setup
# --------------------------------------------------------------------------

def init_database():
    """
    Create the database and tables if they do not exist yet.

    schema.sql is the single source of truth; we simply replay it. Every
    statement is CREATE ... IF NOT EXISTS, so this is safe to run on boot.
    """
    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        raw_sql = handle.read()

    # Strip "-- ..." comment lines so the naive split on ";" stays correct.
    cleaned = re.sub(r"^\s*--.*$", "", raw_sql, flags=re.MULTILINE)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]

    # Connect without selecting a database, because the first statement is the
    # CREATE DATABASE itself.
    conn = get_connection(with_database=False)
    try:
        cursor = conn.cursor()
        for statement in statements:
            cursor.execute(statement)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def check_connection():
    """Return (ok, message) so startup can report a clear error to the console."""
    try:
        conn = get_connection(with_database=False)
        conn.close()
        return True, "MySQL connection OK"
    except mysql.connector.Error as exc:
        return False, f"MySQL connection failed: {exc}"


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def create_dataset(original_filename, stored_path, file_size, n_rows, n_columns, profile):
    return run(
        """
        INSERT INTO datasets
            (original_filename, stored_path, file_size_bytes, n_rows, n_columns, profile_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (original_filename, stored_path, file_size, n_rows, n_columns, json.dumps(profile, default=str)),
    )


def get_dataset(dataset_id):
    row = fetch_one("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
    if row and row.get("profile_json"):
        row["profile"] = json.loads(row["profile_json"])
    return row


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def create_analysis(dataset_id, objective):
    return run(
        "INSERT INTO analyses (dataset_id, objective, status) VALUES (%s, %s, 'queued')",
        (dataset_id, objective),
    )


def get_analysis(analysis_id):
    row = fetch_one("SELECT * FROM analyses WHERE id = %s", (analysis_id,))
    if row and row.get("results_json"):
        row["results"] = json.loads(row["results_json"])
    return row


def update_analysis(analysis_id, **fields):
    """Update any subset of columns on an analysis row."""
    if not fields:
        return
    allowed = {
        "status", "problem_type", "target_column", "evaluation_metric",
        "best_model", "best_score", "objective_status", "objective_reason",
        "ai_source", "error_message", "results_json", "completed_at",
    }
    updates, values = [], []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Refusing to update unknown column: {key}")
        updates.append(f"{key} = %s")
        values.append(value)
    values.append(analysis_id)
    run(f"UPDATE analyses SET {', '.join(updates)} WHERE id = %s", tuple(values))


def save_results(analysis_id, results):
    run(
        "UPDATE analyses SET results_json = %s WHERE id = %s",
        (json.dumps(results, default=str), analysis_id),
    )


def list_recent_analyses(limit=10):
    return fetch_all(
        """
        SELECT a.id, a.objective, a.status, a.problem_type, a.best_model,
               a.best_score, a.objective_status, a.created_at,
               d.original_filename
        FROM analyses a
        JOIN datasets d ON d.id = a.dataset_id
        ORDER BY a.id DESC
        LIMIT %s
        """,
        (limit,),
    )


# --------------------------------------------------------------------------
# Progress steps
# --------------------------------------------------------------------------

def create_steps(analysis_id, steps):
    """steps is a list of (step_key, step_label) tuples in display order."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO analysis_steps (analysis_id, step_order, step_key, step_label)
            VALUES (%s, %s, %s, %s)
            """,
            [(analysis_id, i, key, label) for i, (key, label) in enumerate(steps)],
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def start_step(analysis_id, step_key):
    run(
        """
        UPDATE analysis_steps
        SET status = 'running', started_at = NOW()
        WHERE analysis_id = %s AND step_key = %s
        """,
        (analysis_id, step_key),
    )


def finish_step(analysis_id, step_key, detail=None, status="done"):
    run(
        """
        UPDATE analysis_steps
        SET status = %s, detail = %s, finished_at = NOW()
        WHERE analysis_id = %s AND step_key = %s
        """,
        (status, detail, analysis_id, step_key),
    )


def get_steps(analysis_id):
    return fetch_all(
        "SELECT * FROM analysis_steps WHERE analysis_id = %s ORDER BY step_order",
        (analysis_id,),
    )


# --------------------------------------------------------------------------
# Model results
# --------------------------------------------------------------------------

def save_model_result(analysis_id, result):
    run(
        """
        INSERT INTO model_results
            (analysis_id, model_name, status, train_time_sec, primary_metric,
             primary_score, metrics_json, params_json, error_message, is_best)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            analysis_id,
            result.get("model_name"),
            result.get("status", "trained"),
            result.get("train_time_sec"),
            result.get("primary_metric"),
            result.get("primary_score"),
            json.dumps(result.get("metrics", {}), default=str),
            json.dumps(result.get("params", {}), default=str),
            result.get("error_message"),
            1 if result.get("is_best") else 0,
        ),
    )


def get_model_results(analysis_id):
    rows = fetch_all(
        "SELECT * FROM model_results WHERE analysis_id = %s ORDER BY primary_score DESC",
        (analysis_id,),
    )
    for row in rows:
        row["metrics"] = json.loads(row["metrics_json"]) if row.get("metrics_json") else {}
        row["params"] = json.loads(row["params_json"]) if row.get("params_json") else {}
    return rows


# --------------------------------------------------------------------------
# Reports and users
# --------------------------------------------------------------------------

def save_report(analysis_id, report_md):
    existing = fetch_one("SELECT id FROM reports WHERE analysis_id = %s", (analysis_id,))
    if existing:
        run("UPDATE reports SET report_md = %s, generated_at = NOW() WHERE id = %s",
            (report_md, existing["id"]))
    else:
        run("INSERT INTO reports (analysis_id, report_md) VALUES (%s, %s)",
            (analysis_id, report_md))


def get_report(analysis_id):
    return fetch_one("SELECT * FROM reports WHERE analysis_id = %s", (analysis_id,))


def get_user_by_email(email):
    return fetch_one("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))


def create_user(email, password_hash):
    return run(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
        (email.lower().strip(), password_hash),
    )
