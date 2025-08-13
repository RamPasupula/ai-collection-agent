
import sqlite3
import pandas as pd

DB = "customers.db"
CSV_FILE = "customers.csv" 

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    missed_payment INTEGER NOT NULL DEFAULT 0, -- 0 or 1
    best_call_start TEXT NOT NULL, -- "HH:MM"
    best_call_end TEXT NOT NULL,   -- "HH:MM"
    best_call_day TEXT,            -- e.g., "Monday" or "Weekend"
    timezone TEXT NOT NULL,         -- IANA tz e.g. "America/New_York"
    dst_observed INTEGER NOT NULL DEFAULT 1, -- 0 or 1
    how_long INTEGER NOT NULL,
    State TEXT NOT NULL,
    last_contact_date TEXT,         -- YYYY-MM-DD
    preferred_contact_method TEXT,  -- phone, sms, email
    call_attempts_count INTEGER DEFAULT 0,
    successful_contact_rate REAL DEFAULT 0.0,
    amount_due REAL DEFAULT 0.0,
    days_past_due INTEGER DEFAULT 0,
    risk_score INTEGER DEFAULT 0
);
"""

SAMPLE_INSERTS = [
    ("Amy Gonzales", "(771)967-7351", 1, "15:30", "22:15", "Saturday",
     "America/Los_Angeles", 0, 88, "IA", "2025-08-05", "phone", 15, 0.37, 4775.10, 347, 16),

    ("Kerry Reyes", "+1-052-074-2414x7865", 1, "15:45", "08:45", "Tuesday",
     "Asia/Kolkata", 0, 85, "MI", "2025-02-15", "phone", 0, 0.71, 2024.37, 115, 3),

    ("Craig Henderson", "181-924-5450", 0, "18:00", "22:30", "Monday",
     "America/New_York", 0, 76, "MN", "2025-01-04", "sms", 3, 0.48, 259.30, 332, 90),

    ("Patricia White", "611-740-2092", 0, "12:15", "19:30", "Thursday",
     "Europe/London", 0, 54, "GA", "2025-04-09", "email", 10, 0.69, 1458.99, 205, 12),

    ("Thomas Evans", "+1-568-743-2626x399", 1, "10:45", "15:15", "Wednesday",
     "America/New_York", 1, 45, "TX", "2025-05-21", "phone", 4, 0.80, 3849.45, 122, 47),

    ("Brittany Morgan", "104.400.7258", 0, "09:15", "13:45", "Friday",
     "Australia/Sydney", 1, 39, "WA", "2025-06-28", "sms", 1, 0.15, 742.55, 7, 20),

    ("William Hall", "+1-427-220-3219", 1, "14:00", "20:30", "Sunday",
     "America/Los_Angeles", 1, 66, "CA", "2025-03-12", "phone", 8, 0.58, 2245.70, 88, 55),

    ("Sophia Brown", "876-202-3381x262", 0, "11:45", "18:15", "Weekend",
     "Asia/Kolkata", 1, 51, "NJ", "2025-07-14", "email", 2, 0.33, 1190.20, 145, 9),

    ("Daniel Carter", "(830)720-1720", 1, "17:30", "21:00", "Saturday",
     "Europe/London", 0, 73, "FL", "2025-01-23", "sms", 12, 0.66, 3720.00, 256, 40),

    ("Emily Scott", "604-611-2963", 0, "08:30", "14:45", "Monday",
     "America/New_York", 0, 28, "NY", "2025-03-30", "phone", 6, 0.22, 990.75, 42, 25)
]


def setup():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(CREATE_SQL)
    # insert if empty
    # cur.execute("SELECT COUNT(*) FROM customers")
    # if cur.fetchone()[0] == 0:
    #     cur.executemany(
    #         "INSERT INTO customers (name, phone, missed_payment, best_call_start, best_call_end, timezone) VALUES (?, ?, ?, ?, ?, ?)",
    #         SAMPLE_INSERTS
    #     )
    #     conn.commit()
    #     print("Inserted sample rows.")
    # else:
    #     print("Table already has rows; skipping inserts.")
    conn.close()
    
def import_csv():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_csv(CSV_FILE, nrows=200)
        # Replace NaNs with None for SQLite compatibility
        df = df.where(pd.notnull(df), None)
        # Append data to table
        df.to_sql("customers", conn, if_exists="append", index=False)
        conn.commit()
        print(f"✅ Imported {len(df)} rows from {CSV_FILE}")
    except FileNotFoundError:
        print(f"⚠️ CSV file '{CSV_FILE}' not found. Skipping import.")
    finally:
        conn.close()
    

if __name__ == "__main__":
    setup()
    import_csv()



