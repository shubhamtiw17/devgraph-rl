package com.devgraph.shapes;
import com.devgraph.shapes.Utils;
public abstract class Shape {
    protected String color;
    public Shape(String color) { this.color = color; }
    public abstract double area();
    public String describe() {
        return String.format("%s(color=%s, area=%.2f)", getClass().getSimpleName(), color, area());
    }
}
