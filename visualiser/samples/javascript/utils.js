export function validatePositive(value, name = 'value') {
  if (value <= 0) throw new Error(`${name} must be positive, got ${value}`);
  return value;
}
export function roundResult(value, digits = 4) {
  return parseFloat(value.toFixed(digits));
}
