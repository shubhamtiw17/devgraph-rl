"""Utility helpers shared across the shapes package."""

def validate_positive(value: float, name: str = "value") -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def round_result(value: float, digits: int = 4) -> float:
    return round(value, digits)
