"""Area calculator."""
from src.shapes.shapes import Circle, Rectangle, Triangle
from src.shapes.utils import round_result

class AreaCalculator:
    def __init__(self) -> None:
        self._shapes: list = []
    def add_circle(self, radius: float) -> None:
        self._shapes.append(Circle(radius))
    def add_rectangle(self, width: float, height: float) -> None:
        self._shapes.append(Rectangle(width, height))
    def add_triangle(self, base: float, height: float) -> None:
        self._shapes.append(Triangle(base, height))
    def total_area(self) -> float:
        return round_result(sum(s.area() for s in self._shapes))
    def largest(self):
        return max(self._shapes, key=lambda s: s.area(), default=None)
    def report(self) -> str:
        lines = [s.describe() for s in self._shapes]
        lines.append(f"Total area: {self.total_area()}")
        return "\n".join(lines)
