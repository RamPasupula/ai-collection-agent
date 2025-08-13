import sqlite3
import random

class CustomerUpdater:
    def __init__(self, db_path="customers.db"):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _record_exists(self, conn, name, phone):
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM customers WHERE name = ? OR phone = ?", (name, phone))
        return cursor.fetchone() is not None

    def add_or_update_customer(self, name, phone):
        """Insert a new customer or update missing fields if they already exist."""
        conn = self._connect()
        cursor = conn.cursor()

        if self._record_exists(conn, name, phone):
            print(f"✅ Customer '{name}' exists. Updating missing fields...")
            cursor.execute("""
                UPDATE customers
                SET timezone = COALESCE(timezone, ?),
                    best_call_start = COALESCE(best_call_start, ?),
                    best_call_end = COALESCE(best_call_end, ?),
                    best_call_day = COALESCE(best_call_day, ?),
                    amount_due = COALESCE(amount_due, ?),
                    days_past_due = COALESCE(days_past_due, ?),
                    risk_score = COALESCE(risk_score, ?),
                    missed_payment =1,
                    how_long = COALESCE(how_long, ?)
                WHERE name = ? OR phone = ?
            """, (
                self._random_timezone(),
                "09:00",
                "17:00",
                random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                round(random.uniform(50, 500), 2),
                random.randint(0, 90),
                random.randint(1, 10),
                self._random_duration(),
                name,
                phone
            ))
        else:
            print(f"➕ Adding new customer: {name}")
            cursor.execute("""
                INSERT INTO customers (
                    name, phone, timezone, best_call_start, best_call_end,
                    best_call_day, amount_due, days_past_due, risk_score, how_long, State, missed_payment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            """, (
                name,
                phone,
                self._random_timezone(),
                "09:00",
                "17:00",
                random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                round(random.uniform(50, 500), 2),
                random.randint(0, 90),
                random.randint(1, 10),
                self._random_duration(),
                self._random_state(),
                1  # missed_payment set to 1 for new customers
            ))

        conn.commit()
        conn.close()

    def _random_timezone(self):
        return random.choice(["EST"])
    
    def _random_state(self):
        return random.choice(["NY", "CA", "TX", "FL", "IL", "GA", "VA", "WA", "PA", "OH"])


    def _random_duration(self):
        return random.choice(["16", "1", "5", "10", "3" "7", "2", "4", "6", "8", "12"  ])

if __name__ == "__main__":
    
    updater = CustomerUpdater()

    customers_to_add = [
        ("Venkat", "7746410174"),
        ("Raj", "7325168337"),
        ("Prasad", "5712252310"),  
        ("Ram Pasupula", "+19048555482")
    ]

    for name, phone in customers_to_add:
        updater.add_or_update_customer(name, phone)

    print("✅ Customer update process completed.")
