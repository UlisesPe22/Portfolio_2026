import sqlite3
import time
import json
import requests
import random
import sys
from datetime import datetime, timedelta
import os


API_URL = os.environ.get("HIVE_API_URL", "http://127.0.0.1:5000/api/hive")
DB_PATH = os.environ.get("HIVE_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "hive_data.db"))

def read_rows_from_db(db_path, offset=0, limit=167):
    """
    Reads a specified number of rows from the database, starting from an offset.
    Returns a list of dictionaries.
    """ 
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT time, hive_number, hive_status, hive_temp, hive_humidity, hive_pressure FROM hive_data LIMIT ? OFFSET ?",
        (limit, offset)
    )
    
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None
    return [dict(row) for row in rows]

def send_row(row):
    """
    Sends a single row to the API endpoint.
    Returns the response object.
    """
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(API_URL, json=row, headers=headers, timeout=10)
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return resp
    except requests.exceptions.RequestException as e:
        print(f"Failed to POST to API: {e}", file=sys.stderr)
        return None

def main(delay=0.5):
    """
    Main function to orchestrate the test.
    """

    rows = read_rows_from_db(DB_PATH, offset=0, limit=287)
    
    if rows is None:
        print("DB not found or no rows found after offset. Exiting.")
        return
    # To ensure data types are correct for JSON serialization, we'll convert them
    # before sending.
    for row in rows:
        for key, value in row.items():
            if isinstance(value, datetime):
                row[key] = value.isoformat()
    
    print(f"Streaming {len(rows)} rows to {API_URL} with {delay}s delay between sends.")
    
    for r in rows:

        payload = {
            "time": r.get("time"),
            "hive_number": r.get("hive_number"),
            "hive_status": r.get("hive_status"),
            "hive_temp": r.get("hive_temp"),
            "hive_humidity": r.get("hive_humidity"),
            "hive_pressure": r.get("hive_pressure"),
        }
        
        try:
            resp = send_row(payload)
            if resp:
                print("\n--- NEW RECORD SENT ---")
                print(f"Payload: {payload}")
                print(f"Status Code: {resp.status_code}")
                
                # We expect the API to return the prediction in its response JSON
                resp_json = resp.json()
                
                print("API Response:", json.dumps(resp_json, indent=2))
                
                # Check for the model's output in the response
                if "prediction" in resp_json:
                    print("SUCCESS: Model prediction found in response!")
                else:
                    print("WARNING: 'prediction' not found in response.")
        
        except Exception as e:
            print("POST failed:", e)
        
        time.sleep(delay)

if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        print("Dry run mode: No data will be sent to the API.")
        rows = read_rows_from_db(DB_PATH, offset=0, limit=287)
        if rows:
            print("Payloads to be sent:")
            for r in rows:
                print(r)
    else:
        main(delay=0.5)
