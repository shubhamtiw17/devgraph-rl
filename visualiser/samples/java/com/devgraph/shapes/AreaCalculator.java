package com.devgraph.shapes;
import com.devgraph.shapes.Circle;
import com.devgraph.shapes.Shape;
import com.devgraph.shapes.Utils;
import java.util.ArrayList;
public class AreaCalculator {
    private ArrayList<Shape> shapes = new ArrayList<>();
    public void addCircle(double radius) { shapes.add(new Circle(radius, "white")); }
    public double totalArea() { return shapes.stream().mapToDouble(Shape::area).sum(); }
    public String report() {
        StringBuilder sb = new StringBuilder();
        for (Shape s : shapes) sb.append(s.describe()).append("\n");
        sb.append("Total: ").append(Utils.roundResult(totalArea(), 4));
        return sb.toString();
    }
}
