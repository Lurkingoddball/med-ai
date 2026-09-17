import os
import sqlite3
import json
import time
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "analytics.db")

def get_db_connection():
    """Returns a SQLite connection to analytics.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_analytics_db():
    """Initializes the analytics SQLite database schema if not already present."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            ip_address TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            search_keywords TEXT,
            selected_scope TEXT,
            sources_cited TEXT,
            response_length INTEGER,
            latency_seconds REAL,
            user_agent TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(
    session_id: str,
    ip_address: str,
    user_query: str,
    search_keywords: str,
    selected_scope: str,
    sources_cited: list,
    response_length: int,
    latency_seconds: float,
    user_agent: str = "Unknown"
):
    """Logs a single user interaction into the analytics database."""
    try:
        init_analytics_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sources_json = json.dumps(sources_cited, ensure_ascii=False)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO query_logs (
                session_id, ip_address, timestamp, user_query,
                search_keywords, selected_scope, sources_cited,
                response_length, latency_seconds, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            ip_address if ip_address else "127.0.0.1",
            current_time,
            user_query,
            search_keywords,
            selected_scope,
            sources_json,
            response_length,
            round(latency_seconds, 2),
            user_agent
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging interaction: {e}")

def get_all_logs() -> pd.DataFrame:
    """Returns all query logs as a pandas DataFrame."""
    init_analytics_db()
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM query_logs ORDER BY id DESC", conn)
    conn.close()
    return df

def get_analytics_metrics():
    """Computes summary KPI metrics for the creator dashboard."""
    init_analytics_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total queries
    cursor.execute("SELECT COUNT(*) FROM query_logs")
    total_queries = cursor.fetchone()[0]

    # Unique IPs
    cursor.execute("SELECT COUNT(DISTINCT ip_address) FROM query_logs")
    unique_ips = cursor.fetchone()[0]

    # Average latency
    cursor.execute("SELECT AVG(latency_seconds) FROM query_logs")
    avg_latency_row = cursor.fetchone()[0]
    avg_latency = round(avg_latency_row, 2) if avg_latency_row else 0.0

    # Top cited book
    cursor.execute("SELECT sources_cited FROM query_logs WHERE sources_cited IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    book_counts = {}
    for r in rows:
        try:
            sources = json.loads(r[0])
            for s in sources:
                book = s.get("book", "Unknown")
                book_counts[book] = book_counts.get(book, 0) + 1
        except Exception:
            continue

    top_book = max(book_counts, key=book_counts.get) if book_counts else "None"
    top_book_clean = top_book.split(" - ")[0] if " - " in top_book else top_book

    return {
        "total_queries": total_queries,
        "unique_ips": unique_ips,
        "avg_latency": avg_latency,
        "top_book": top_book_clean,
        "book_counts": book_counts
    }

def clear_logs():
    """Clears all logs (admin maintenance)."""
    init_analytics_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM query_logs")
    conn.commit()
    conn.close()
