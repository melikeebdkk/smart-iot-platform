import json
import paho.mqtt.client as mqtt

STATE = "OFF"

def on_message(client, userdata, msg):
    global STATE
    payload = json.loads(msg.payload.decode())
    STATE = payload.get("device_state", "OFF")
    print("💡 Light is now", STATE, "| payload:", payload)

client = mqtt.Client()
client.connect("localhost", 1883)
client.subscribe("iot/commands/light")
client.on_message = on_message

print("💡 Light simulator running... listening on iot/commands/light")
client.loop_forever()
