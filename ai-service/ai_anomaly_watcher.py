import time
import json
import psycopg2
import paho.mqtt.client as mqtt
from datetime import datetime

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TELEMETRY_TOPIC = "iot/telemetry/#"
LIGHT_TOPIC = "iot/commands/light"

DB = dict(
    host="127.0.0.1",
    port=5432,
    dbname="iotdb",
    user="iotuser",
    password="iotpass"
)

POWER_THRESHOLD = 8000   # W üstü anomali

def get_conn():
    return psycopg2.connect(**DB)

def send_light_off():
    payload = {
        "device_id": "light_01",
        "parent_device": "home_01",
        "device_state": "OFF"
    }
    mqtt_client.publish(LIGHT_TOPIC, json.dumps(payload))
    print("💡 Light OFF sent")

def insert_notification(device, power):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ai_notifications (device_id, message, level)
        VALUES (%s, %s, 'CRITICAL')
    """, (
        device,
        f"⚠️ High power detected: {power}W"
    ))
    conn.commit()
    cur.close()
    conn.close()
    print("🧠 Notification saved")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        power = data.get("power", 0)
        device = data.get("device_id", "unknown")

        print(f"📡 {device} → {power}W")

        if power > POWER_THRESHOLD:
            print("🔥 ANOMALY DETECTED")
            insert_notification(device, power)
            send_light_off()
    except Exception as e:
        print("ERR:", e)

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
mqtt_client.subscribe(TELEMETRY_TOPIC)

print("🧠 AI Anomaly Watcher ACTIVE")
mqtt_client.loop_forever()

