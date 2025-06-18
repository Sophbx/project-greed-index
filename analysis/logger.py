import os
from datetime import datetime

def log_metrics(metrics: dict, strategy_name: str = "Unnamed Strategy", logs_dir: str = "logs"):
    """
    Logs evaluation metrics to a timestamped log file.
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{strategy_name.replace(' ', '_').lower()}_log.txt")

    with open(log_path, 'a') as f:
        f.write(f"\n--- {strategy_name} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for k, v in metrics.items():
            f.write(f"{k.replace('_', ' ').capitalize()}: {v:.4f}\n")
        f.write("-" * 40 + "\n")

    print(f"Metrics logged to {log_path}")
