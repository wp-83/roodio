def get_mood_label(v, a):
    if a >= 0.5:
        return "MD-0000001" if v >= 0.5 else "MD-0000004"
    else:
        return "MD-0000003" if v >= 0.5 else "MD-0000002"

# MD-0000001 => happy
# MD-0000002 => sad
# MD-0000003 => relaxed
# MD-0000004 => angry