def is_anomaly(confidence, threshold=0.7):
    """Flag as anomaly if confidence is too low."""
    return confidence < threshold

def check_alert(pred_class, confidence, threshold=0.7):
    """
    Takes predicted class + confidence.
    Returns (is_danger, reason) — used by app.py.
    """
    DISTRESS_CLASSES = {
        7: "Unknown Anomaly",
        8: "Fall Detected",
        9: "Panic Run",
        10: "Struggling",
        11: "Sudden Freeze"
    }

    if pred_class in DISTRESS_CLASSES:
        return True, f"Distress class detected: {DISTRESS_CLASSES[pred_class]}"

    if confidence < threshold:
        return True, f"Low confidence: {confidence*100:.1f}% (threshold: {threshold*100:.0f}%)"

    return False, "Normal activity detected"