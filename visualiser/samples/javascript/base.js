import { validatePositive } from './utils.js';
export class Shape {
  constructor(color = 'white') { this.color = color; }
  area() { throw new Error('area() not implemented'); }
  describe() {
    return `${this.constructor.name}(color=${this.color}, area=${this.area().toFixed(2)})`;
  }
}
