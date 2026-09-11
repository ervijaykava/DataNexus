"""
Application configuration.

Everything the app needs to know about its environment lives here, read once
from the .env file. Keeping it in one small module means there is exactly one
place to look when something is misconfigured.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths ----

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ---------------------------------------------------------------- mysql ----

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "data_scientist_agent")

# --------------------------------------------------------------- gemini ----

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# The app is designed to keep working without Gemini: if the key is missing or
# the API call fails, a local rule-based analyst produces the same structured
# recommendation. This flag is only used for display in the UI.
GEMINI_ENABLED = bool(GEMINI_API_KEY)

# ---------------------------------------------------------------- limits ---

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Datasets larger than this are sampled before model training so that the
# analysis finishes in a reasonable time. The full dataset is still profiled.
MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "50000"))

# How many rows of real data are shown to Gemini. We never send the whole file.
GEMINI_SAMPLE_ROWS = 8

ALLOWED_EXTENSIONS = {".csv"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
