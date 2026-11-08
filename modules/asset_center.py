"""Asset registry and maintenance workspace for PowerLine AI.

This module stores operational asset/task metadata separately from AI detections.
It does not infer physical damage from component detections.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime
from modules.inspection_history import DB_PATH

def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def initialize_asset_tables():
    with _connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            location TEXT DEFAULT '',
            latitude REAL,
            longitude REAL,
            voltage_level TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'Medium',
            status TEXT NOT NULL DEFAULT 'Open',
            due_date TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE SET NULL
        )""")

def list_assets():
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM assets ORDER BY id DESC")]

def add_asset(asset_code, name, location="", latitude=None, longitude=None, voltage_level="", notes=""):
    with _connect() as conn:
        cur=conn.execute("""INSERT INTO assets
            (asset_code,name,location,latitude,longitude,voltage_level,notes,created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (asset_code.strip(),name.strip(),location.strip(),latitude,longitude,
             voltage_level.strip(),notes.strip(),datetime.now().astimezone().isoformat(timespec="seconds")))
        return cur.lastrowid

def delete_asset(asset_id):
    with _connect() as conn:
        conn.execute("DELETE FROM assets WHERE id=?", (int(asset_id),))

def list_tasks():
    with _connect() as conn:
        return [dict(r) for r in conn.execute("""SELECT t.*, a.asset_code, a.name AS asset_name
            FROM maintenance_tasks t LEFT JOIN assets a ON a.id=t.asset_id
            ORDER BY CASE t.priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END,
            t.id DESC""")]

def add_task(asset_id, title, description="", priority="Medium", due_date=""):
    with _connect() as conn:
        cur=conn.execute("""INSERT INTO maintenance_tasks
            (asset_id,title,description,priority,status,due_date,created_at)
            VALUES (?,?,?,?, 'Open', ?, ?)""",
            (asset_id,title.strip(),description.strip(),priority,due_date,
             datetime.now().astimezone().isoformat(timespec="seconds")))
        return cur.lastrowid

def update_task_status(task_id, status):
    if status not in {"Open","In Progress","Completed","Cancelled"}:
        raise ValueError("Unsupported task status")
    with _connect() as conn:
        conn.execute("UPDATE maintenance_tasks SET status=? WHERE id=?", (status,int(task_id)))

def asset_summary():
    with _connect() as conn:
        assets=conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        open_tasks=conn.execute("SELECT COUNT(*) FROM maintenance_tasks WHERE status IN ('Open','In Progress')").fetchone()[0]
        high_tasks=conn.execute("SELECT COUNT(*) FROM maintenance_tasks WHERE priority='High' AND status IN ('Open','In Progress')").fetchone()[0]
    return {"assets":assets,"open_tasks":open_tasks,"high_priority_tasks":high_tasks}
