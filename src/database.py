import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CONFIG = {
    "dbname": "skill_gap_db",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

# Global flag so the app knows if DB is available
DB_AVAILABLE = False

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        return None

def init_db():
    global DB_AVAILABLE
    conn = get_db_connection()
    if not conn:
        print("⚠️ WARNING: Database unavailable. App will run but results won't be saved.")
        DB_AVAILABLE = False
        return
    
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    try:
        cursor.execute("ALTER TABLE users DROP COLUMN IF EXISTS password_hash;")
    except Exception:
        pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_skills (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        skill_name VARCHAR(100) NOT NULL,
        verified_score INTEGER CHECK (verified_score >= 0 AND verified_score <= 5),
        UNIQUE(user_id, skill_name)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        recommended_role VARCHAR(255) NOT NULL,
        missing_skills TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    DB_AVAILABLE = True
    print("✅ Database connected and tables ready!")

def get_or_create_user(email):
    if not DB_AVAILABLE:
        print("⚠️ Skipping user creation - DB unavailable")
        return None
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
    if not DB_AVAILABLE or not user_id:
        print("⚠️ Skipping skill save - DB unavailable")
        return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()

    for skill, score in verified_scores.items():
        cursor.execute("""
            INSERT INTO user_skills (user_id, skill_name, verified_score)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, skill_name)
            DO UPDATE SET verified_score = EXCLUDED.verified_score;
        """, (user_id, skill, score))

    conn.commit()
    cursor.close()
    conn.close()

def save_recommendation(user_id, best_job, missing_skills_list):
    if not DB_AVAILABLE or not user_id:
        print("⚠️ Skipping recommendation save - DB unavailable")
        return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()

    missing_skills_str = ", ".join(missing_skills_list)
    cursor.execute("""
        INSERT INTO recommendations (user_id, recommended_role, missing_skills)
        VALUES (%s, %s, %s)
    """, (user_id, best_job, missing_skills_str))

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_db()