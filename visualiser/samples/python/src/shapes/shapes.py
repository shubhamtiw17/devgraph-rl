"""Concrete shape implementations."""
import math
from src.shapes.base import Shape
from src.shapes.utils import validate_positive, round_result

class Circle(Shape):
    def __init__(self, radius: float, color: str = "white") -> None:
        super().__init__(color)
        self.radius = validate_positive(radius, "radius")
    def area(self) -> float:
        return round_result(math.pi * self.radius ** 2)
    def circumference(self) -> float:
        return round_result(2 * math.pi * self.radius)

class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: str = "white") -> None:
        super().__init__(color)
        self.width  = validate_positive(width,  "width")
        self.height = validate_positive(height, "height")
    def area(self) -> float:
        return round_result(self.width * self.height)
    def perimeter(self) -> float:
        return round_result(2 * (self.width + self.height))

class Triangle(Shape):
    def __init__(self, base: float, height: float, color: str = "white") -> None:
        super().__init__(color)
        self.base   = validate_positive(base,   "base")
        self.height = validate_positive(height, "height")
    def area(self) -> float:
        return round_result(0.5 * self.base * self.height)
