function findOddOccurringNumbers(arr) {
    let xor = 0;

    // Step 1: XOR all elements. Result is XOR of two odd occurring elements
    for (let num of arr) {
        xor ^= num;
    }

    // Step 2: Find a set bit in the result (there must be at least one)
    let setBit = xor & -xor;

    let num1 = 0, num2 = 0;

    // Step 3: Divide all numbers into two groups and XOR separately
    for (let num of arr) {
        if ((num & setBit) === 0) {
            num1 ^= num;
        } else {
            num2 ^= num;
        }
    }

    return [num1, num2];
}
