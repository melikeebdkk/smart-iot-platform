import json
import time
import random
import paho.mqtt.client as mqtt
from datetime import datetime

# MQTT broker bilgileri
BROKER = "localhost"
PORT = 1883
TOPIC = "iot/telemetry/device_01"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

print("Simulated IoT device started...")

DEVICES = {
    "smart_plug_klima":  (800, 2200),
    "smart_plug_light":  (50, 200),
    "smart_plug_fridge": (100, 400),
    "smart_plug_tv":     (80, 300),
    "smart_plug_other":  (30, 150),
}

PARENT_DEVICE = "home_01"

while True:
    for device_id, (min_p, max_p) in DEVICES.items():

        power = round(random.uniform(min_p, max_p), 2)

        # %5 ihtimalle anomali
        if random.random() < 0.05:
            power = round(random.uniform(max_p * 1.3, max_p * 1.8), 2)

        payload = {
            "device_id": device_id,
            "parent_device": PARENT_DEVICE,
            "timestamp": datetime.utcnow().isoformat(),
            "power": power,
            "device_state": "ON"
        }

        topic = f"iot/telemetry/{device_id}"
        client.publish(topic, json.dumps(payload))

        print("Published:", payload)

    time.sleep(2)

