#include "utils.h"
#include <stdexcept>
#include <cmath>
#include <string>
double validatePositive(double value, const char* name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    return value;
}
double roundResult(double value, int digits) {
    double factor = std::pow(10.0, digits);
    return std::round(value * factor) / factor;
}
