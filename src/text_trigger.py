DISTRESS_KEYWORDS = [
    "help", "save me", "danger", "emergency", "attack",
    "scared", "stop it", "leave me alone", "call police",
    "someone help", "i'm scared", "please help", "i need help",
    "somebody help", "let me go", "i'm in danger", "don't hurt me"
]

def check_distress_text(text):
    """Check text for distress keywords. Returns (flag, matched_word)."""
    text_lower = text.lower()
    for keyword in DISTRESS_KEYWORDS:
        if keyword in text_lower:
            return True, keyword
    return False, None