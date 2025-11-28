import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import pandas as pd

# Get the project root directory (parent of dashboard folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "stress_detection.db")

def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Stress records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stress_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            rf_prediction INTEGER NOT NULL,
            rf_status TEXT NOT NULL,
            rf_confidence REAL,
            lr_prediction INTEGER NOT NULL,
            lr_status TEXT NOT NULL,
            lr_confidence REAL,
            hr_value REAL,
            temperature_value REAL,
            accel_x REAL,
            accel_y REAL,
            accel_z REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, timestamp)
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stress_records_user_id 
        ON stress_records(user_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stress_records_timestamp 
        ON stress_records(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stress_records_date 
        ON stress_records(date(timestamp))
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, email: str, password: str) -> Tuple[bool, Optional[str]]:
    """Create a new user. Returns (success, error_message)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        """, (username, email, password_hash))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        elif "email" in str(e):
            return False, "Email already exists"
        return False, "User creation failed"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user. Returns user dict if successful, None otherwise"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute("""
        SELECT id, username, email, created_at
        FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'created_at': user[3]
        }
    return None

def save_stress_record(user_id: int, timestamp: str, rf_prediction: int, rf_status: str,
                      rf_confidence: float, lr_prediction: int, lr_status: str,
                      lr_confidence: float, hr_value: float = None, temperature_value: float = None,
                      accel_x: float = None, accel_y: float = None, accel_z: float = None):
    """Save a stress record to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO stress_records 
            (user_id, timestamp, rf_prediction, rf_status, rf_confidence,
             lr_prediction, lr_status, lr_confidence, hr_value, temperature_value,
             accel_x, accel_y, accel_z)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, timestamp, rf_prediction, rf_status, rf_confidence,
              lr_prediction, lr_status, lr_confidence, hr_value, temperature_value,
              accel_x, accel_y, accel_z))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving stress record: {e}")
        return False
    finally:
        conn.close()

def get_user_stress_records(user_id: int, limit: int = 100) -> List[Dict]:
    """Get stress records for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM stress_records
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    
    records = cursor.fetchall()
    conn.close()
    
    return [dict(record) for record in records]

def get_daily_stress_count(user_id: int, date: str = None) -> List[Dict]:
    """Get daily stress count for a user. If date is None, returns all days"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if date:
        cursor.execute("""
            SELECT 
                date(timestamp) as day,
                COUNT(*) as stress_count,
                SUM(CASE WHEN rf_prediction > 0 OR lr_prediction > 0 THEN 1 ELSE 0 END) as stressed_count
            FROM stress_records
            WHERE user_id = ? AND date(timestamp) = ?
            GROUP BY date(timestamp)
            ORDER BY day DESC
        """, (user_id, date))
    else:
        cursor.execute("""
            SELECT 
                date(timestamp) as day,
                COUNT(*) as total_records,
                SUM(CASE WHEN rf_prediction > 0 OR lr_prediction > 0 THEN 1 ELSE 0 END) as stressed_count,
                SUM(CASE WHEN rf_prediction = 0 AND lr_prediction = 0 THEN 1 ELSE 0 END) as normal_count
            FROM stress_records
            WHERE user_id = ?
            GROUP BY date(timestamp)
            ORDER BY day DESC
            LIMIT 30
        """, (user_id,))
    
    records = cursor.fetchall()
    conn.close()
    
    return [dict(record) for record in records]

def get_stress_statistics(user_id: int) -> Dict:
    """Get overall stress statistics for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total records
    cursor.execute("""
        SELECT COUNT(*) as total FROM stress_records WHERE user_id = ?
    """, (user_id,))
    total = cursor.fetchone()[0]
    
    # Stressed records
    cursor.execute("""
        SELECT COUNT(*) as stressed FROM stress_records 
        WHERE user_id = ? AND (rf_prediction > 0 OR lr_prediction > 0)
    """, (user_id,))
    stressed = cursor.fetchone()[0]
    
    # Today's records
    cursor.execute("""
        SELECT COUNT(*) as today_total FROM stress_records 
        WHERE user_id = ? AND date(timestamp) = date('now')
    """, (user_id,))
    today_total = cursor.fetchone()[0]
    
    # Today's stressed
    cursor.execute("""
        SELECT COUNT(*) as today_stressed FROM stress_records 
        WHERE user_id = ? AND date(timestamp) = date('now') 
        AND (rf_prediction > 0 OR lr_prediction > 0)
    """, (user_id,))
    today_stressed = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_records': total,
        'total_stressed': stressed,
        'stress_percentage': (stressed / total * 100) if total > 0 else 0,
        'today_total': today_total,
        'today_stressed': today_stressed,
        'today_stress_percentage': (today_stressed / today_total * 100) if today_total > 0 else 0
    }

# Initialize database on import
init_database()

