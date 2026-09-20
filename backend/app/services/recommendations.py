def generate_recommendation(pattern_label: str, peak_period: dict, avg_value: float) -> str:
    """
    Generates actionable energy-saving recommendations only when appropriate.
    If consumption is low and balanced, returns a concise efficiency confirmation.
    """
    peak_start = peak_period.get("start", "18:00")
    peak_end = peak_period.get("end", "20:00")
    peak_val = peak_period.get("value", 0.0)

    if "High" in pattern_label or peak_val >= 200.0:
        return f"Shift non-essential high-power loads (EV charging, dishwasher, washing machine) outside the peak period ({peak_start} - {peak_end}) to minimize peak demand charges."
    elif "Medium" in pattern_label or peak_val >= 120.0:
        return f"Moderate consumption detected. Stagger appliance usage during the peak window ({peak_start} - {peak_end}) to prevent transitioning into peak tier rates."
    else:
        return "Consumption pattern is currently within optimal energy efficiency thresholds. No immediate load shifting required."
