import sqlite3
import os
from datetime import datetime

# Store DB in a writable location on HF Spaces
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', 'data', 'skill_gap.db')
DB_PATH  = os.path.abspath(DB_PATH)

DB_AVAILABLE = False

def get_db_connection():
    try:
        # 1. First, try to get the Neon Connection String from Render's environment
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            # If it exists (like on Render), connect to Neon!
            conn = psycopg2.connect(db_url)
        else:
            # 2. If it doesn't exist, fall back to localhost for local testing
            conn = psycopg2.connect(**DB_CONFIG)
            
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def init_db():
    global DB_AVAILABLE
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️  WARNING: Database unavailable. App will run but results won't be saved.")
            DB_AVAILABLE = False
            return
        cursor = conn.cursor()

        # Users
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Per-skill scores
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_skills (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER REFERENCES users(id),
            skill_name     TEXT NOT NULL,
            verified_score INTEGER,
            assessed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Recommendations history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER REFERENCES users(id),
            recommended_role TEXT NOT NULL,
            missing_skills   TEXT,
            score_pct        INTEGER DEFAULT 0,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit()
        conn.close()
        DB_AVAILABLE = True
        print("✅ SQLite database ready!")
    except Exception as e:
        print(f"⚠️  DB init error: {e}")
        DB_AVAILABLE = False

def get_or_create_user(email):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    if user:
        user_id = user[0]
    else:
        cursor.execute("INSERT INTO users (email) VALUES (?)", (email,))
        user_id = cursor.lastrowid
        conn.commit()
    conn.close()
    return user_id

def save_verified_skills(user_id, verified_scores):
    if not DB_AVAILABLE or not user_id: return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    for skill, score in verified_scores.items():
        cursor.execute("""
            INSERT INTO user_skills (user_id, skill_name, verified_score, assessed_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, skill, score, now))
    conn.commit()
    conn.close()

def save_recommendation(user_id, best_job, missing_skills_list, score_pct=0):
    if not DB_AVAILABLE or not user_id: return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recommendations (user_id, recommended_role, missing_skills, score_pct, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, best_job, ", ".join(missing_skills_list), score_pct, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_user_dashboard(email):
    conn = get_db_connection()
    if not conn: return None, None, None

    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None, None, None
    user_id = row[0]

    # Recommendation history
    cursor.execute("""
        SELECT recommended_role, score_pct, missing_skills, created_at
        FROM recommendations
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (user_id,))
    rec_history = cursor.fetchall()

    # Latest skill scores (one per skill, most recent)
    cursor.execute("""
        SELECT skill_name, verified_score, assessed_at
        FROM user_skills
        WHERE user_id = ?
        GROUP BY skill_name
        HAVING assessed_at = MAX(assessed_at)
        ORDER BY skill_name
    """, (user_id,))
    latest_skills = cursor.fetchall()

    # Progress rows
    cursor.execute("""
        SELECT skill_name, verified_score, assessed_at
        FROM user_skills
        WHERE user_id = ?
        ORDER BY skill_name, assessed_at DESC
    """, (user_id,))
    progress_rows = cursor.fetchall()

    conn.close()
    return rec_history, latest_skills, progress_rows

if __name__ == "__main__":
    init_db()