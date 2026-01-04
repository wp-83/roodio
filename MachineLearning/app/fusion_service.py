def get_mood_label(v, a):
    if a >= 0.5:
        return "happy" if v >= 0.5 else "angry"
    else:
        return "relaxed" if v >= 0.5 else "sad"
