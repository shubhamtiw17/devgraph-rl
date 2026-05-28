"""Base shape class."""
from src.shapes.utils import validate_positive

class Shape:
    def __init__(self, color: str = "white") -> None:
        self.color = color
    def area(self) -> float:
        raise NotImplementedError
    def describe(self) -> str:
        return f"{self.__class__.__name__}(color={self.color}, area={self.area():.2f})"
