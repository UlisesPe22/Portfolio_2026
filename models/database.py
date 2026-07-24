# hive_app/models/database.py
import sqlite3
from flask import current_app, g

def get_db():
    """
    Return a per-request sqlite3.Connection stored in flask.g.
    """
    if "db" not in g:
        db_path = current_app.config.get("DATABASE")
        # detect types allows storing proper types; row_factory returns sqlite3.Row
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_app(app):
    """
    Register teardown and ensure the hive_data table exists.
    Call this from app factory (app.py).
    """
    app.teardown_appcontext(close_db)
    # Ensure table exists
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                hive_number INTEGER NOT NULL,
                hive_status TEXT NOT NULL,
                hive_temp REAL NOT NULL,
                hive_humidity REAL NOT NULL,
                hive_pressure REAL NOT NULL
            )
        """)
        db.commit()
