from datetime import datetime, timedelta

from app.db import get_db_connection
from app.anomaly_rules import check_power_anomaly


def get_recent_power_records(device_id: str, minutes: int = 10):
    """
    DB'den son X dakika içindeki power kayıtlarını çeker
    """
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT time, power
        FROM telemetry
        WHERE device_id = %s
          AND time >= NOW() - INTERVAL %s
        ORDER BY time ASC
    """

    cur.execute(query, (device_id, f"{minutes} minutes"))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def check_recent_anomalies(device_id: str, minutes: int = 10):
    """
    Son X dakikadaki anomalileri bulur
    """
    records = get_recent_power_records(device_id, minutes)

    anomalies = []

    for ts, power in records:
        result = check_power_anomaly(power)

        if result["is_anomaly"]:
            anomalies.append({
                "timestamp": ts.isoformat(),
                "power": power,
                "reason": result["reason"]
            })

    return anomalies


# --- BASİT TEST ---
if __name__ == "__main__":
    anomalies = check_recent_anomalies(
        device_id="device_01",
        minutes=30
    )

    print("Bulunan anomaliler:")
    for a in anomalies:
        print(a)
