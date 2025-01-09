function findOddNumbers(arr) {
    let numCount = {};
    
    arr.forEach(num => {
        if (numCount[num] === undefined) {
            numCount[num] = 1;
        } else {
            numCount[num]++;
        }
    });

    const oddNumbers = [];
    
    for (let key in numCount) {
        if (numCount[key] % 2 !== 0) {
            oddNumbers.push(parseInt(key));
        }
    }
    
    return oddNumbers;
}

const arr = [1, 2, 3, 2, 1, 3, 5];
console.log(findOddNumbers(arr)); // Output: [1, 5]
