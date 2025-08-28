import sqlite3
import hashlib
from datetime import datetime

class StudentDB:
    def __init__(self, db_name="students.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with column checks"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                track TEXT NOT NULL,
                created_at TEXT,
                status TEXT
            )
        ''')
        
        # Check if created_at column exists, if not add it
        cursor.execute("PRAGMA table_info(students)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'created_at' not in columns:
            cursor.execute("ALTER TABLE students ADD COLUMN created_at TEXT")
        
        if 'status' not in columns:
            cursor.execute("ALTER TABLE students ADD COLUMN status TEXT DEFAULT 'active'")
        
        # Update existing records with default values if needed
        cursor.execute("UPDATE students SET created_at = ? WHERE created_at IS NULL", 
                      (datetime.now().isoformat(),))
        cursor.execute("UPDATE students SET status = 'active' WHERE status IS NULL")
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def add_student(self, name, email, password, track):
        """Add new student to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            hashed_password = self.hash_password(password)
            created_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO students (name, email, password, track, created_at, status) VALUES (?, ?, ?, ?, ?, ?)",
                (name, email, hashed_password, track, created_at, 'active')
            )
            
            conn.commit()
            student_id = cursor.lastrowid
            conn.close()
            
            return self.get_student(student_id)
            
        except sqlite3.IntegrityError:
            raise ValueError("Email already exists")
    
    def verify_login(self, email, password):
        """Verify student login credentials"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        hashed_password = self.hash_password(password)
        
        cursor.execute(
            "SELECT id, name, email, track, created_at, status FROM students WHERE email = ? AND password = ?",
            (email, hashed_password)
        )
        
        student = cursor.fetchone()
        conn.close()
        
        if student:
            return {
                "id": student[0],
                "name": student[1],
                "email": student[2],
                "track": student[3],
                "created_at": student[4] or datetime.now().isoformat(),
                "status": student[5] or 'active'
            }
        return None
    
    def get_all_students(self):
        """Get all students from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, email, track, created_at, status FROM students")
        students = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": s[0],
                "name": s[1],
                "email": s[2],
                "track": s[3],
                "created_at": s[4] or "Unknown",
                "status": s[5] or 'active'
            }
            for s in students
        ]
    
    def get_student(self, student_id):
        """Get student by ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, email, track, created_at, status FROM students WHERE id = ?",
            (student_id,)
        )
        
        student = cursor.fetchone()
        conn.close()
        
        if student:
            return {
                "id": student[0],
                "name": student[1],
                "email": student[2],
                "track": student[3],
                "created_at": student[4] or "Unknown",
                "status": student[5] or 'active'
            }
        return None
    
    def update_student(self, student_id, **kwargs):
        """Update student information"""
        allowed_fields = ["name", "track", "status"]
        update_fields = []
        values = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = ?")
                values.append(value)
        
        if not update_fields:
            return None
        
        values.append(student_id)
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute(
            f"UPDATE students SET {', '.join(update_fields)} WHERE id = ?",
            values
        )
        
        conn.commit()
        conn.close()
        
        return self.get_student(student_id)
    
    def delete_student(self, student_id):
        """Delete student from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        
        return deleted

# Create database instance
db = StudentDB()