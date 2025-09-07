from kafka import KafkaConsumer

# Debug: Print connection attempt
print("🔄 Connecting to Kafka...")

# Create Kafka Consumer
consumer = KafkaConsumer(
    'lstm_topic',  # Topic name
    bootstrap_servers='localhost:9092',  # Kafka broker
    auto_offset_reset='earliest',  # Start from the beginning
    enable_auto_commit=True,
    group_id='lstm_consumer_group'
)

# Debug: Connection success
print("✅ Connected! Listening for messages...")

# Start consuming messages
for message in consumer:
    print(f"📡 Received message: {message.value.decode('utf-8')}")
