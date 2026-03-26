from datetime import datetime

def alert(triggered, reason="", severity="HIGH"):
    """Log alert to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if triggered:
        with open("alerts.log", "a") as f:
            f.write(f"[{timestamp}] ALERT - {severity} - {reason}\n")