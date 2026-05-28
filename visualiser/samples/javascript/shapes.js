import { Shape } from './base.js';
import { validatePositive, roundResult } from './utils.js';
export class Circle extends Shape {
  constructor(radius, color = 'white') {
    super(color);
    this.radius = validatePositive(radius, 'radius');
  }
  area() { return roundResult(Math.PI * this.radius ** 2); }
  circumference() { return roundResult(2 * Math.PI * this.radius); }
}
export class Rectangle extends Shape {
  constructor(width, height, color = 'white') {
    super(color);
    this.width  = validatePositive(width,  'width');
    this.height = validatePositive(height, 'height');
  }
  area() { return roundResult(this.width * this.height); }
  perimeter() { return roundResult(2 * (this.width + this.height)); }
}
export class Triangle extends Shape {
  constructor(base, height, color = 'white') {
    super(color);
    this.base   = validatePositive(base,   'base');
    this.height = validatePositive(height, 'height');
  }
  area() { return roundResult(0.5 * this.base * this.height); }
}
