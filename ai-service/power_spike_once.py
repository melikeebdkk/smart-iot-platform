import json
import random
import paho.mqtt.client as mqtt
from datetime import datetime,timezone

BROKER = "localhost"
TOPIC = "iot/telemetry/power"

client = mqtt.Client()
client.connect(BROKER, 1883)

power = random.randint(3000, 5000)

payload = {
    "device_id": "smart_plug_klima",
    "parent_device": "home_01",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "power": 3500,
    "device_state": "ON"
}


client.publish("iot/telemetry/power", json.dumps(payload))
print("💥 POWER SPIKE gönderildi:", power, "W")

client.disconnect()

