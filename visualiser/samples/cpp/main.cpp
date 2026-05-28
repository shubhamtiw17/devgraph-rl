#include "calculator.h"
#include <iostream>
int main() {
    AreaCalculator calc;
    calc.addCircle(5.0);
    calc.addRectangle(4.0, 6.0);
    std::cout << calc.report() << std::endl;
    return 0;
}
