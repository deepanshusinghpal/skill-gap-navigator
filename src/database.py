import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CONFIG = {
    "dbname":   "skill_gap_db",
    "user":     "postgres",
    "password": os.getenv("DB_PASSWORD", "Db@123"),
    "host":     "localhost",
    "port":     "5432"
}

DB_AVAILABLE = False

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def init_db():
    global DB_AVAILABLE
    conn = get_db_connection()
    if not conn:
        print("⚠️  WARNING: Database unavailable. App will run but results won't be saved.")
        DB_AVAILABLE = False
        return
    cursor = conn.cursor()

    # Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id         SERIAL PRIMARY KEY,
        email      VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")

    # Per-skill scores — now with assessed_at timestamp for history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_skills (
        id             SERIAL PRIMARY KEY,
        user_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,
        skill_name     VARCHAR(100) NOT NULL,
        verified_score INTEGER CHECK (verified_score >= 0 AND verified_score <= 5),
        assessed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")

    # Recommendations history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id               SERIAL PRIMARY KEY,
        user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
        recommended_role VARCHAR(255) NOT NULL,
        missing_skills   TEXT,
        score_pct        INTEGER DEFAULT 0,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")

    # Add missing columns to existing tables safely
    for col, definition in [
        ("assessed_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("score_pct",   "INTEGER DEFAULT 0"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE user_skills ADD COLUMN IF NOT EXISTS assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
            cursor.execute(f"ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS score_pct INTEGER DEFAULT 0;")
        except:
            pass

    conn.commit()
    cursor.close()
    conn.close()
    DB_AVAILABLE = True
    print("✅ Database connected and tables ready!")

def get_or_create_user(email):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user:
        user_id = user[0]
    else:
        cursor.execute("INSERT INTO users (email) VALUES (%s) RETURNING id", (email,))
        user_id = cursor.fetchone()[0]
        conn.commit()
    cursor.close()
    conn.close()
    return user_id

def save_verified_skills(user_id, verified_scores):
    if not DB_AVAILABLE or not user_id: return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    # Insert a NEW row each time (for history tracking)
    for skill, score in verified_scores.items():
        cursor.execute("""
            INSERT INTO user_skills (user_id, skill_name, verified_score, assessed_at)
            VALUES (%s, %s, %s, NOW())
        """, (user_id, skill, score))
    conn.commit()
    cursor.close()
    conn.close()

def save_recommendation(user_id, best_job, missing_skills_list, score_pct=0):
    if not DB_AVAILABLE or not user_id: return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recommendations (user_id, recommended_role, missing_skills, score_pct, created_at)
        VALUES (%s, %s, %s, %s, NOW())
    """, (user_id, best_job, ", ".join(missing_skills_list), score_pct))
    conn.commit()
    cursor.close()
    conn.close()

# ── Dashboard queries ────────────────────────────────────────────
def get_user_dashboard(email):
    """Returns full history for a user: assessments + skill scores over time."""
    conn = get_db_connection()
    if not conn: return None, None, None

    cursor = conn.cursor()

    # Get user id
    cursor.execute("SELECT id, created_at FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if not row:
        cursor.close(); conn.close()
        return None, None, None
    user_id, joined = row

    # Recommendation history
    cursor.execute("""
        SELECT recommended_role, score_pct, missing_skills, created_at
        FROM recommendations
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 10
    """, (user_id,))
    rec_history = cursor.fetchall()

    # Latest skill scores
    cursor.execute("""
        SELECT DISTINCT ON (skill_name) skill_name, verified_score, assessed_at
        FROM user_skills
        WHERE user_id = %s
        ORDER BY skill_name, assessed_at DESC
    """, (user_id,))
    latest_skills = cursor.fetchall()

    # Skill progress — first vs last score per skill
    cursor.execute("""
        SELECT skill_name,
               FIRST_VALUE(verified_score) OVER (PARTITION BY skill_name ORDER BY assessed_at ASC) AS first_score,
               FIRST_VALUE(verified_score) OVER (PARTITION BY skill_name ORDER BY assessed_at DESC) AS last_score
        FROM user_skills
        WHERE user_id = %s
    """, (user_id,))
    progress_rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rec_history, latest_skills, progress_rows

if __name__ == "__main__":
    init_db()