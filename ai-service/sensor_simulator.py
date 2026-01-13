import json
import time
import math
import random
import paho.mqtt.client as mqtt
from datetime import datetime
MQTT_HOST = "localhost"
MQTT_PORT = 1883

TOPIC = "iot/telemetry/temp_hum"

PARENT_DEVICE = "home_01"
DEVICE_ID = "temp_hum_sensor_01"

client = mqtt.Client()
client.connect(MQTT_HOST, MQTT_PORT, 60)

base_temp = 24.0
base_hum = 50.0

print("✅ Sensor simulator started. Publishing to:", TOPIC)

while True:
    temp = base_temp + math.sin(time.time() / 120) * 1.5 + random.uniform(-0.2, 0.2)
    hum = base_hum + math.sin(time.time() / 180) * 6 + random.uniform(-1.0, 1.0)

    payload = {
    "device_id": DEVICE_ID,
    "parent_device": PARENT_DEVICE,
    "timestamp": datetime.utcnow().isoformat(),
    "temperature": round(temp, 2),
    "humidity": round(hum, 1),
    "power": 0,
    "device_state": "SENSOR"
}

    client.publish(TOPIC, json.dumps(payload))
    print("📤", payload)

    time.sleep(10)
