def clock(seconds: float) -> str:
    """308 -> '00:05:08'"""
    total = round(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def human(seconds: float) -> str:
    """702 -> '11m 42s'"""
    total = round(seconds)
    hours, minutes, secs = total // 3600, total % 3600 // 60, total % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"
