function sumOddNumbers(array) {
    // Using the reduce method to sum the odd numbers in the array
    return array.reduce((sum, number) => {
        // If number is odd add it to sum, else add zero
        return sum + (number % 2 !== 0 ? number : 0);
    }, 0);
}

// Testing the function with given example
console.log(sumOddNumbers([1, 2, 3, 4, 5])); // Output: 9
