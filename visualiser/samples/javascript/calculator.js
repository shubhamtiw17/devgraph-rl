import { Circle, Rectangle, Triangle } from './shapes.js';
import { roundResult } from './utils.js';
export class AreaCalculator {
  constructor() { this._shapes = []; }
  addCircle(radius)           { this._shapes.push(new Circle(radius)); }
  addRectangle(width, height) { this._shapes.push(new Rectangle(width, height)); }
  addTriangle(base, height)   { this._shapes.push(new Triangle(base, height)); }
  totalArea() { return roundResult(this._shapes.reduce((s, sh) => s + sh.area(), 0)); }
  largest()   { return this._shapes.reduce((a, b) => a.area() > b.area() ? a : b, null); }
  report()    { return [...this._shapes.map(s => s.describe()), `Total: ${this.totalArea()}`].join('\n'); }
}
