function sumOfOddNumbers(arr) {
    // Initialize sum to accumulate only odd numbers
    let sum = 0;
    
    // Iterate through each number in the array
    for (let num of arr) {
        // Check if the number is odd
        if (num % 2 !== 0) {
            // Add the odd number to the sum
            sum += num;
        }
    }
    
    // Return the total sum of all odd numbers
    return sum;
}

// Example usage:
console.log(sumOfOddNumbers([1, 2, 3, 4, 5])); // Output: 9
