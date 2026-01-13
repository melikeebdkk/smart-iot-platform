import time
import psycopg2
import statistics
import json
import paho.mqtt.client as mqtt

# ---------- MQTT ----------
MQTT_HOST = "localhost"
MQTT_PORT = 1883
LIGHT_TOPIC = "iot/commands/light"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)

# ---------- DB ----------
DB = dict(
    host="127.0.0.1",
    port=5432,
    dbname="iotdb",
    user="iotuser",
    password="iotpass"
)

WINDOW_MIN = 30
POLL_SEC = 10
Z_THRESHOLD = 3.0

# ---------- MQTT ACTION ----------
def turn_light_off():
    payload = {
        "device_id": "light_01",
        "parent_device": "home_01",
        "device_state": "OFF",
        "source": "AI"
    }
    mqtt_client.publish(LIGHT_TOPIC, json.dumps(payload))
    print("💡 AI ışıkları kapattı!")

# ---------- DB ----------
def get_recent_powers(conn, device_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT power
            FROM telemetry
            WHERE device_id = %s
              AND time >= NOW() - INTERVAL '30 minutes'
              AND power IS NOT NULL
            ORDER BY time ASC
        """, (device_id,))
        return [r[0] for r in cur.fetchall()]

def get_latest(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT time, device_id, power
            FROM telemetry
            WHERE power IS NOT NULL
            ORDER BY time DESC
            LIMIT 1
        """)
        return cur.fetchone()

def insert_alert(conn, device_id, power, mean, std):
    z = abs(power - mean) / std if std > 0 else 0

    payload = {
        "power": power,
        "mean": mean,
        "std": std,
        "z": z,
        "model": "z-score"
    }

    cur = conn.cursor()

    # 1️⃣ Açık bir POWER_ANOMALY var mı?
    cur.execute("""
        SELECT id
        FROM ai_notifications
        WHERE device_id = %s
          AND type = 'POWER_ANOMALY'
          AND status IN ('OPEN', 'ACK')
        ORDER BY time DESC
        LIMIT 1
    """, (device_id,))

    row = cur.fetchone()

    if row:
        # 2️⃣ Varsa → UPDATE
        alert_id = row[0]
        cur.execute("""
            UPDATE ai_notifications
            SET time = now(),
                payload = %s
            WHERE id = %s
        """, (json.dumps(payload), alert_id))

        print(f"🔁 UPDATED existing POWER_ANOMALY for {device_id}")

    else:
        # 3️⃣ Yoksa → INSERT
        cur.execute("""
            INSERT INTO ai_notifications (device_id, type, severity, payload, source)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            device_id,
            "POWER_ANOMALY",
            "HIGH",
            json.dumps(payload),
            "AI_WATCHER"
        ))

        print(f"🆕 NEW POWER_ANOMALY for {device_id}")

    conn.commit()



# ---------- MAIN LOOP ----------
print("🧠 AI Anomaly Watcher (Auto Action) running...")

conn = psycopg2.connect(**DB)

while True:
    try:
        row = get_latest(conn)
        if not row:
            time.sleep(POLL_SEC)
            continue

        ts, device_id, power = row

        history = get_recent_powers(conn, device_id)
        if len(history) < 10:
            time.sleep(POLL_SEC)
            continue

        mean = statistics.mean(history)
        std = statistics.stdev(history)

        z = abs(power - mean) / std if std > 0 else 0

        if z > Z_THRESHOLD:
            print(f"🚨 ANOMALY DETECTED on {device_id} → {power:.1f}W (z={z:.2f})")
            insert_alert(conn, device_id, power, mean, std)
            turn_light_off()
        else:
            print(f"OK {device_id}: {power:.1f}W  z={z:.2f}")

    except Exception as e:
        print("AI ERROR:", e)

    time.sleep(POLL_SEC)

