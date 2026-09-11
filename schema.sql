-- =========================================================================
-- DATA SCIENTIST AGENT 2.0 - MySQL schema
--
-- Run this once:  mysql -u root -p < schema.sql
-- (The application also creates these tables automatically on startup.)
--
-- Design note: uploaded CSV files are kept on the server filesystem and only
-- their metadata and path are stored here. The database holds the analysis
-- trail - what the agent decided and why - not the raw data.
-- =========================================================================

CREATE DATABASE IF NOT EXISTS data_scientist_agent
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE data_scientist_agent;


-- Accounts are only needed to download a report. Analysis itself is public.
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    email         VARCHAR(190) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- One row per uploaded file.
CREATE TABLE IF NOT EXISTS datasets (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_path       VARCHAR(500) NOT NULL,
    file_size_bytes   BIGINT NOT NULL DEFAULT 0,
    n_rows            INT NOT NULL DEFAULT 0,
    n_columns         INT NOT NULL DEFAULT 0,
    profile_json      LONGTEXT,
    uploaded_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- One row per analysis run.
CREATE TABLE IF NOT EXISTS analyses (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id        INT NOT NULL,
    objective         TEXT NOT NULL,
    -- queued | running | needs_target | completed | failed
    status            VARCHAR(30) NOT NULL DEFAULT 'queued',
    problem_type      VARCHAR(50),
    target_column     VARCHAR(190),
    evaluation_metric VARCHAR(50),
    best_model        VARCHAR(100),
    best_score        DOUBLE,
    -- FULFILLED | PARTIALLY FULFILLED | NOT FULFILLED
    objective_status  VARCHAR(30),
    objective_reason  TEXT,
    ai_source         VARCHAR(30),          -- gemini | local-fallback
    error_message     TEXT,
    results_json      LONGTEXT,             -- full dashboard payload
    created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at      DATETIME,
    CONSTRAINT fk_analyses_dataset
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    INDEX idx_analyses_status (status)
) ENGINE=InnoDB;


-- The live progress trail shown on the analysis screen. Statuses are written
-- by the pipeline as each step actually starts and finishes - never faked.
CREATE TABLE IF NOT EXISTS analysis_steps (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT NOT NULL,
    step_order  INT NOT NULL,
    step_key    VARCHAR(60) NOT NULL,
    step_label  VARCHAR(190) NOT NULL,
    -- pending | running | done | failed | skipped
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    detail      TEXT,
    started_at  DATETIME,
    finished_at DATETIME,
    CONSTRAINT fk_steps_analysis
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    INDEX idx_steps_analysis (analysis_id, step_order)
) ENGINE=InnoDB;


-- One row per model the agent attempted, including the ones that failed.
CREATE TABLE IF NOT EXISTS model_results (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id     INT NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'trained',  -- trained | failed
    train_time_sec  DOUBLE,
    primary_metric  VARCHAR(50),
    primary_score   DOUBLE,
    metrics_json    LONGTEXT,
    params_json     LONGTEXT,
    error_message   TEXT,
    is_best         TINYINT(1) NOT NULL DEFAULT 0,
    CONSTRAINT fk_models_analysis
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    INDEX idx_models_analysis (analysis_id)
) ENGINE=InnoDB;


-- The generated report, stored so it can be re-read without re-running.
CREATE TABLE IF NOT EXISTS reports (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id  INT NOT NULL,
    report_md    LONGTEXT,
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_reports_analysis
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    INDEX idx_reports_analysis (analysis_id)
) ENGINE=InnoDB;
