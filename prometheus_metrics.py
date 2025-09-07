from prometheus_client import Gauge, start_http_server
import psutil
import time

# Define Prometheus metrics
g_cpu = Gauge('cpu_usage', 'CPU Usage')
g_mem = Gauge('memory_usage', 'Memory Usage')
g_prediction = Gauge('lstm_prediction', 'Latest LSTM Prediction')

def update_metrics(prediction):
    """Updates Prometheus metrics."""
    g_cpu.set(psutil.cpu_percent())
    g_mem.set(psutil.virtual_memory().percent)
    g_prediction.set(prediction)

# Start HTTP server for Prometheus
start_http_server(9090)

while True:
    time.sleep(2)  # Update metrics every 2 seconds
