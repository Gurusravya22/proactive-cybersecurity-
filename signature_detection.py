# signature_detection.py
def signature_detection(data, signatures):
    detected = []
    for entry in data:
        for category, sigs in signatures.items():
            if entry in sigs:
                detected.append(f"Detected: {category}")
                break
        else:
            detected.append("No Match")
    return detected
