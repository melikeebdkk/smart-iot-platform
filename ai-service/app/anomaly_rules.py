def check_power_anomaly(power: float):
    """
    Basit kural tabanlı anomali kontrolü
    """

    lower_threshold = 50
    upper_threshold = 150

    if power < lower_threshold:
        return {
            "power": power,
            "is_anomaly": True,
            "reason": "power below lower threshold"
        }

    if power > upper_threshold:
        return {
            "power": power,
            "is_anomaly": True,
            "reason": "power above upper threshold"
        }

    return {
        "power": power,
        "is_anomaly": False,
        "reason": "power within normal range"
    }


# --- BASİT TEST ---
if __name__ == "__main__":
    test_values = [30, 90, 180]

    for v in test_values:
        print(check_power_anomaly(v))
