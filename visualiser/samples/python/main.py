import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

from src.shapes.calculator import AreaCalculator

def main() -> None:
    calc = AreaCalculator()
    calc.add_circle(5.0)
    calc.add_rectangle(4.0, 6.0)
    calc.add_triangle(3.0, 8.0)
    print(calc.report())
    largest = calc.largest()
    if largest:
        print(f"Largest: {largest.describe()}")

if __name__ == "__main__":
    main()
