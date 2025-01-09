function sumOddNumbers(arr) {
    let sum = 0; // initialize sum to 0
    for(let i = 0; i < arr.length; i++) { // iterate over array
        if (arr[i] % 2 !== 0) { // check if number is odd
            sum += arr[i]; // add to sum if number is odd
        }
    }
    return sum; // return the sum
}
