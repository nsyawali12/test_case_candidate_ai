function sumOfOddNumbers(arr) {
    return arr.reduce((acc, curr) => {
        return acc + curr;
    }, 0);
}

const numbers = [1, 2, 3, 4, 5];
const result = sumOfOddNumbers(numbers.filter(num => num % 2 !== 0));
result;
