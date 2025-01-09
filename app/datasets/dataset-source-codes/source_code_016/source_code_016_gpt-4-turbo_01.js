function sumOfOddNumbers(numbers) {
    // Function to add all odd numbers in the provided array
    return numbers.reduce((acc, num) => {
        // Check if the number is odd, if so add it to the accumulator
        if (num % 2 !== 0) {
            acc += num;
        }
        return acc;
    }, 0);
}

// Example usage:
const result = sumOfOddNumbers([1, 2, 3, 4, 5]);
console.log(result);  // Output will be 9
