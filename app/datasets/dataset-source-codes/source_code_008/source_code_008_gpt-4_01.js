// This function will take in an array as a parameter and return the maximum number
function maxNumberFromArray(numbers){
  return Math.max(...numbers);
}

// Usage:
console.log(maxNumberFromArray([1, 5, 9, 3, 7])); // This will output: 9
console.log(maxNumberFromArray([-2, 0, -5, -1])); // This will output: 0
