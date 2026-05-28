package com.devgraph.shapes;
import com.devgraph.shapes.Shape;
import com.devgraph.shapes.Utils;
public class Circle extends Shape {
    private double radius;
    public Circle(double radius, String color) {
        super(color);
        this.radius = Utils.validatePositive(radius, "radius");
    }
    public double area() { return Utils.roundResult(Math.PI * radius * radius, 4); }
    public double circumference() { return Utils.roundResult(2 * Math.PI * radius, 4); }
}
