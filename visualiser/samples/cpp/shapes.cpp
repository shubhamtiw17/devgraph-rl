#include "shapes.h"
#include "utils.h"
#include <cmath>
Circle::Circle(double radius, std::string color) : Shape(color) {
    this->radius = validatePositive(radius, "radius");
}
double Circle::area() const { return roundResult(M_PI * radius * radius); }
double Circle::circumference() const { return roundResult(2 * M_PI * radius); }
Rectangle::Rectangle(double width, double height, std::string color) : Shape(color) {
    this->width  = validatePositive(width,  "width");
    this->height = validatePositive(height, "height");
}
double Rectangle::area() const { return roundResult(width * height); }
double Rectangle::perimeter() const { return roundResult(2 * (width + height)); }
