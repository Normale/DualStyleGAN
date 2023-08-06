import json
import base64
from kafka import KafkaProducer
from PIL import Image

# Set up Kafka producer
topic_name = "style-transfer"
bootstrap_servers = "localhost:19092"
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda m: json.dumps(m).encode("utf-8"),
)

image_path = "./data/content/081680.jpg"
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Encode image data to base64
content_bytes = base64.b64encode(image_data).decode("utf-8")

# Create a JSON message with the required arguments
message = {
    # "content": "./data/content/006750.jpg",
    # "content": "./data/content/019706.jpg",
    # "content": "./data/content/077417.jpg",
    "content": "./data/content/1.jpg",
    "base64content": content_bytes,
    "style": "rpg2",
    "style_id": 53,
    "truncation": 0.75,
    "weight": [0.75] * 7 + [1] * 11,
    "name": "cartoon_transfer",
    "preserve_color": False,
    "model_path": "./checkpoint/",
    "model_name": "generator-001100.pt",
    "output_path": "./output/",
    "data_path": "./data/",
    "align_face": False,
    "exstyle_name": None,
    "wplus": False,
}
print("Sending message to Kafka topic:", topic_name)
# Send the message to the Kafka topic
producer.send(topic_name, message)
producer.flush()

print("Message sent!")