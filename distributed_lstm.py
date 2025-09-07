import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from prometheus_client import start_http_server, Gauge
import psutil
import time
import threading
import json
from kafka import KafkaProducer

# Start Prometheus monitoring
g_cpu = Gauge('cpu_usage', 'CPU Usage')
g_mem = Gauge('memory_usage', 'Memory Usage')

def start_metrics_server():
    """Starts Prometheus metrics server to monitor CPU and Memory usage."""
    start_http_server(9090)
    while True:
        g_cpu.set(psutil.cpu_percent())
        g_mem.set(psutil.virtual_memory().percent)
        time.sleep(2)

# Run Prometheus in a separate thread
threading.Thread(target=start_metrics_server, daemon=True).start()

def create_lstm_model(input_shape):
    """Creates an LSTM model with improved architecture."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(data):
    """Trains an LSTM model on the given time-series data."""
    if len(data) < 51:
        raise ValueError(f"Not enough data to train the LSTM model. At least 51 rows required, but found {len(data)}.")

    X_train, y_train = [], []
    for i in range(50, len(data)):
        X_train.append(data[i-50:i])
        y_train.append(data[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    
    # Save the model
    model.save("lstm_model.h5")
    print("✅ Model saved as lstm_model.h5")
    return model

def send_to_kafka(topic, message):
    """Sends predictions to Kafka"""
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    producer.send(topic, {"prediction": message})
    producer.flush()
    producer.close()
    print(f"📡 Sent prediction to Kafka: {message}")  # Add this line for debugging

def main():
    """Main execution function for LSTM training and monitoring."""
    # Start Spark session
    spark = SparkSession.builder.appName("LSTMTraining").getOrCreate()

    # Load time series data
    file_path = "DataSummary.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Display data types before conversion
        print("🧐 Column types before conversion:")
        print(df.dtypes)

        # Identify numeric columns manually
        valid_columns = ['Train', 'Test', 'Length', 'ED (w=0)', 'DTW (w=100)', 'Default rate']
        numeric_columns = [col for col in valid_columns if col in df.columns]

        if not numeric_columns:
            raise ValueError("❌ No valid numerical time-series columns found in dataset!")

        # Apply numeric conversion only to selected numeric columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Print NaN counts per column before dropping
        print("🔍 NaN values per column before dropping:")
        print(df.isna().sum())

        # Drop NaN values only for selected numeric columns
        df.dropna(subset=numeric_columns, inplace=True)

        print(f"📉 Dataset after cleaning: {df.shape[0]} rows remaining.")

        if df.empty:
            raise ValueError("❌ Error: After cleaning, the dataset is empty. Check if valid numeric data is available!")

        # Choose target column
        target_column = numeric_columns[0]  # Pick the first valid time-series column
        print(f"✅ Using column '{target_column}' for training.")

        # Extract time-series data
        data = df[target_column].values.reshape(-1, 1)
        print(f"📈 Selected column '{target_column}' with {len(data)} rows.")

        # Train model
        model = train_lstm_model(data)

        # Send model summary to Kafka
        summary_text = "LSTM Model Trained Successfully!"
        send_to_kafka("lstm_topic", summary_text)

        print(summary_text)
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
    except ValueError as ve:
        print(f"❌ Error: {ve}")

if __name__ == "__main__":
    main()
