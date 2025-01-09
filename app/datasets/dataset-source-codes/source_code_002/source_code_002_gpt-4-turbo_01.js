function findOddOccuringNumbers(arr) {
    let xorResult = 0;

    // Step 1: XOR of all elements gives us XOR of two odd occurring elements
    for (let num of arr) {
        xorResult ^= num;
    }

    // Step 2: Get the rightmost set bit in the xorResult
    let rightmostSetBit = xorResult & -xorResult;

    let num1 = 0, num2 = 0;

    // Step 3: Divide elements into two groups and XOR
    for (let num of arr) {
        if ((num & rightmostSetBit) === 0) {
            num1 ^= num;
        } else {
            num2 ^= num;
        }
    }

    // Return the two odd occurring numbers
    return [num1, num2];
}

// Example usage:
findOddOccuringNumbers([1, 2, 3, 2, 1, 3, 5]);  // Should return [1, 5]
