"""
Kenya Maize Advisory - MySQL chat history
=========================================

The functions the Streamlit app calls to persist farmer profiles and
conversations. They mirror ``RAG_Advisory_Chatbot.ipynb`` section 10 and read
their connection details from the environment, loading a local ``.env`` file
when one is present.

Nothing connects at import time. Every call opens its own connection, so a
missing database degrades to a message in the app instead of a crash.
"""

from __future__ import annotations

import contextlib
import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import mysql.connector

from rag_pipeline import looks_swahili

DB_ENV_KEYS = ("MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")

DATABASE_SQL = [
    """CREATE TABLE IF NOT EXISTS farmers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(120) NOT NULL,
        county VARCHAR(60),
        preferred_lang CHAR(2) NOT NULL DEFAULT 'en',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def db_configured() -> bool:
    return all(os.environ.get(key) is not None for key in DB_ENV_KEYS)


@contextlib.contextmanager
def db_connection():
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


def create_schema() -> None:
    with db_connection() as connection:
        cursor = connection.cursor()
        for statement in DATABASE_SQL:
            cursor.execute(statement)
        cursor.close()


def create_farmer(full_name: str, county: str | None = None, preferred_lang: str = "en") -> int:
    if not full_name or len(full_name) > 120:
        raise ValueError("full_name must be 1-120 characters")
    if preferred_lang not in ("en", "sw"):
        raise ValueError("preferred_lang must be en or sw")
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO farmers (full_name, county, preferred_lang) VALUES (%s, %s, %s)",
            (full_name, county, preferred_lang))
        farmer_id = cursor.lastrowid
        cursor.close()
    return farmer_id


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


def set_feedback(message_id: int, helpful: bool) -> None:
    if not isinstance(helpful, bool):
        raise ValueError("helpful must be True or False")
    with db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE chat_messages SET feedback = %s WHERE id = %s",
                       (1 if helpful else -1, message_id))
        cursor.close()
