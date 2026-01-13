from app.db import get_db_connection


def get_basic_power_insights(device_id: str, hours: int = 24):
    """
    Son X saatlik power verisi için temel istatistikler üretir
    """
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT power
        FROM telemetry
        WHERE device_id = %s
          AND time >= NOW() - INTERVAL %s
          AND power IS NOT NULL
    """

    cur.execute(query, (device_id, f"{hours} hours"))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return {
            "device_id": device_id,
            "range_hours": hours,
            "message": "No data available"
        }

    powers = [r[0] for r in rows]

    avg_power = sum(powers) / len(powers)
    min_power = min(powers)
    max_power = max(powers)

    # Basit toplam tüketim yaklaşımı
    total_consumption = sum(powers)

    return {
        "device_id": device_id,
        "range_hours": hours,
        "avg_power": round(avg_power, 2),
        "min_power": round(min_power, 2),
        "max_power": round(max_power, 2),
        "total_consumption": round(total_consumption, 2),
        "data_points": len(powers)
    }


# --- BASİT TEST ---
if __name__ == "__main__":
    insights = get_basic_power_insights(
        device_id="device_01",
        hours=24
    )

    print("Basic Power Insights:")
    for k, v in insights.items():
        print(f"{k}: {v}")
