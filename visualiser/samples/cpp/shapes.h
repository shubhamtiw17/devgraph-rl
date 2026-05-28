#pragma once
#include "shape.h"
class Circle : public Shape {
public:
    double radius;
    Circle(double radius, std::string color = "white");
    double area() const override;
    double circumference() const;
};
class Rectangle : public Shape {
public:
    double width, height;
    Rectangle(double width, double height, std::string color = "white");
    double area() const override;
    double perimeter() const;
};
