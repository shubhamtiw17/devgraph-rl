#pragma once
#include "shapes.h"
#include "utils.h"
#include <vector>
#include <memory>
#include <string>
class AreaCalculator {
    std::vector<std::unique_ptr<Shape>> shapes;
public:
    void addCircle(double radius);
    void addRectangle(double width, double height);
    double totalArea() const;
    std::string report() const;
};
