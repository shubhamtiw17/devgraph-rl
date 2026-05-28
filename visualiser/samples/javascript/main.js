import { AreaCalculator } from './calculator.js';
function main() {
  const calc = new AreaCalculator();
  calc.addCircle(5.0);
  calc.addRectangle(4.0, 6.0);
  calc.addTriangle(3.0, 8.0);
  console.log(calc.report());
}
main();
