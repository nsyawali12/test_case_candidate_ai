function findMaxNumber(numbers) {
  // Return the maximum number in the array using Math.max and spread operator
  return Math.max(...numbers);
}

// Examples
console.log(findMaxNumber([1, 5, 9, 3, 7])); // Output: 9
console.log(findMaxNumber([-2, 0, -5, -1])); // Output: 0
