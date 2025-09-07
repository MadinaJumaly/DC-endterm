from prometheus_client import start_http_server, Gauge
import psutil
import time

cpu_usage = Gauge('cpu_usage', 'CPU Usage')
memory_usage = Gauge('memory_usage', 'Memory Usage')

def collect_metrics():
    while True:
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        time.sleep(2)

if __name__ == "__main__":
    start_http_server(8000)
    collect_metrics()
