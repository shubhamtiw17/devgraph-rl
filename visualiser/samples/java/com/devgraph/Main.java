package com.devgraph;
import com.devgraph.shapes.AreaCalculator;
public class Main {
    public static void main(String[] args) {
        AreaCalculator calc = new AreaCalculator();
        calc.addCircle(5.0);
        System.out.println(calc.report());
    }
}
