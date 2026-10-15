"""
Kenya Maize Advisory - accounts and MySQL chat history
======================================================

The functions the Streamlit app calls to create farmer accounts, log them in,
and persist conversations. They mirror ``RAG_Advisory_Chatbot.ipynb`` section 12
and read their connection details from the environment, loading a local ``.env``
file when one is present.

Passwords are stored only as salted PBKDF2-HMAC-SHA256 hashes (standard library,
no extra dependency). Nothing connects at import time; every call opens its own
connection so a missing database degrades to a message instead of a crash.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import mysql.connector
except ImportError:
    mysql = None

from rag_pipeline import looks_swahili

DB_ENV_KEYS = ("MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
PBKDF2_ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

DATABASE_SQL = [
    """CREATE TABLE IF NOT EXISTS farmers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(120) NOT NULL,
        email VARCHAR(190) NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        county VARCHAR(60),
        preferred_lang CHAR(2) NOT NULL DEFAULT 'en',
        profile_image VARCHAR(255),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_farmers_email (email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_sessions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        farmer_id INT NOT NULL,
        started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        language CHAR(2) NOT NULL DEFAULT 'en',
        FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_messages (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id INT NOT NULL,
        farmer_id INT NOT NULL,
        role VARCHAR(10) NOT NULL,
        message MEDIUMTEXT NOT NULL,
        language CHAR(2) NOT NULL DEFAULT 'en',
        sources JSON,
        feedback TINYINT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

ACCOUNT_COLUMNS = [
    ("email", "email VARCHAR(190) NULL"),
    ("password_hash", "password_hash VARCHAR(255) NULL"),
    ("profile_image", "profile_image VARCHAR(255) NULL"),
]


def db_configured() -> bool:
    return all(os.environ.get(key) is not None for key in DB_ENV_KEYS)


@contextlib.contextmanager
def db_connection():
    if mysql is None:
        raise ValueError(
            "mysql-connector-python is not installed in the Python running this app. "
            "Start the app with the project virtualenv: .venv/bin/streamlit run app.py")
    missing = [key for key in DB_ENV_KEYS if os.environ.get(key) is None]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    connection = mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ.get("MYSQL_PASSWORD", ""),
        charset="utf8mb4",
        autocommit=False,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table, column))
    return cursor.fetchone()[0] > 0


def _index_exists(cursor, table: str, index: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s",
        (table, index))
    return cursor.fetchone()[0] > 0


def create_schema() -> None:
    with db_connection() as connection:
        cursor = connection.cursor()
        for statement in DATABASE_SQL:
            cursor.execute(statement)
        for column, definition in ACCOUNT_COLUMNS:
            if not _column_exists(cursor, "farmers", column):
                cursor.execute(f"ALTER TABLE farmers ADD COLUMN {definition}")
        if not _index_exists(cursor, "farmers", "uq_farmers_email"):
            cursor.execute("ALTER TABLE farmers ADD UNIQUE KEY uq_farmers_email (email)")
        cursor.close()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        algorithm, iterations, salt_hex, hash_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived.hex(), hash_hex)


def email_exists(email: str) -> bool:
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM farmers WHERE email = %s", (email,))
        count = cursor.fetchone()[0]
        cursor.close()
    return count > 0


def create_account(full_name: str, email: str, password: str, county: str | None = None,
                   preferred_lang: str = "en", profile_image: str | None = None) -> int:
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()
    if not full_name or len(full_name) > 120:
        raise ValueError("Name must be 1-120 characters.")
    if not EMAIL_RE.match(email) or len(email) > 190:
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if preferred_lang not in ("en", "sw"):
        raise ValueError("preferred_lang must be en or sw")
    if email_exists(email):
        raise ValueError("An account with that email already exists.")
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO farmers (full_name, email, password_hash, county, preferred_lang, profile_image) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (full_name, email, hash_password(password), county, preferred_lang, profile_image))
        farmer_id = cursor.lastrowid
        cursor.close()
    return farmer_id


def authenticate(email: str, password: str) -> dict | None:
    email = (email or "").strip().lower()
    with db_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, full_name, email, password_hash, county, preferred_lang, profile_image "
            "FROM farmers WHERE email = %s", (email,))
        row = cursor.fetchone()
        cursor.close()
    if row is None or not verify_password(password, row.pop("password_hash")):
        return None
    return row


def get_farmer(farmer_id: int) -> dict | None:
    with db_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, full_name, email, county, preferred_lang, profile_image, created_at "
            "FROM farmers WHERE id = %s", (farmer_id,))
        row = cursor.fetchone()
        cursor.close()
    return row


def update_profile_image(farmer_id: int, path: str) -> None:
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE farmers SET profile_image = %s WHERE id = %s", (path, farmer_id))
        cursor.close()


def start_session(farmer_id: int, language: str = "en") -> int:
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (farmer_id, language) VALUES (%s, %s)",
            (farmer_id, language))
        session_id = cursor.lastrowid
        cursor.close()
    return session_id


def store_message(farmer_id: int, session_id: int, question: str, result: dict) -> int:
    language = "sw" if looks_swahili(question) else "en"
    sources = json.dumps(result.get("sources", []))
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (session_id, farmer_id, role, message, language, sources) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (session_id, farmer_id, "user", question, language, None))
        cursor.execute(
            "INSERT INTO chat_messages (session_id, farmer_id, role, message, language, sources) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (session_id, farmer_id, "assistant", result["answer"], language, sources))
        assistant_id = cursor.lastrowid
        cursor.close()
    return assistant_id


def read_history(farmer_id: int, session_id: int, limit: int = 100) -> list[dict]:
    with db_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, role, message, language, sources, feedback, created_at "
            "FROM chat_messages WHERE farmer_id = %s AND session_id = %s "
            "ORDER BY id ASC LIMIT %s", (farmer_id, session_id, limit))
        rows = cursor.fetchall()
        cursor.close()
    return rows


def list_sessions(farmer_id: int, limit: int = 20) -> list[dict]:
    with db_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT s.id, s.started_at, s.language, "
            "(SELECT m.message FROM chat_messages m "
            " WHERE m.session_id = s.id AND m.role = 'user' "
            " ORDER BY m.id ASC LIMIT 1) AS topic "
            "FROM chat_sessions s WHERE s.farmer_id = %s "
            "ORDER BY s.started_at DESC, s.id DESC LIMIT %s",
            (farmer_id, limit))
        rows = cursor.fetchall()
        cursor.close()
    return rows


def set_feedback(message_id: int, helpful: bool) -> None:
    if not isinstance(helpful, bool):
        raise ValueError("helpful must be True or False")
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE chat_messages SET feedback = %s WHERE id = %s",
                       (1 if helpful else -1, message_id))
        cursor.close()
