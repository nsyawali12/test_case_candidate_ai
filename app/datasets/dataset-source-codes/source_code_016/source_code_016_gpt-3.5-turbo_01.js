function sumOfOddNumbers(arr) {
    return arr.flat().reduce((sum, num) => num % 2 !== 0 ? sum + num : sum, 0);
}

console.log(sumOfOddNumbers([[1, 2, 3, 4, 5], 9])); // Output: 18
