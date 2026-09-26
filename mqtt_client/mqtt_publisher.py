import paho.mqtt.client as mqtt
from config.config import BROKER, PORT, TOPIC_RESULT

client = mqtt.Client()
client.connect(BROKER, PORT)
client.loop_start()

def send_result(result):
    print("Sending:", result)
    client.publish(TOPIC_RESULT, result)