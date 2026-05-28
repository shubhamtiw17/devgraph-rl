#pragma once
#include "utils.h"
#include <string>
class Shape {
public:
    std::string color;
    Shape(std::string color = "white") : color(color) {}
    virtual double area() const = 0;
    virtual std::string describe() const;
    virtual ~Shape() = default;
};
