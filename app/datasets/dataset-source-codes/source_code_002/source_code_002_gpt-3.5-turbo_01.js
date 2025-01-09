function findOddOccurrences(arr) {
    const occurrences = {};
    
    arr.forEach(num => {
        if (occurrences[num] === undefined) {
            occurrences[num] = 1;
        } else {
            occurrences[num]++;
        }
    });
    
    const oddNumbers = [];
    
    for (let num in occurrences) {
        if (occurrences[num] % 2 !== 0) {
            oddNumbers.push(parseInt(num));
        }
    }
    
    return oddNumbers;
}

// Test the function with the given example
const inputArr = [1, 2, 3, 2, 1, 3, 5];
const result = findOddOccurrences(inputArr);
console.log(result); // Output: [1, 5]
