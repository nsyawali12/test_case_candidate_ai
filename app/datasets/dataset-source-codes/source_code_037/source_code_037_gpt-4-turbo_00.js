const findLongestSubstring = (str) => {
    let start = 0;
    let maxLength = 0;
    let longestSub = '';
    let map = {};

    for (let i = 0; i < str.length; i++) {
        const char = str[i];

        if (map[char] >= start) {
            start = map[char] + 1;
        }

        map[char] = i;

        if (i - start + 1 > maxLength) {
            maxLength = i - start + 1;
            longestSub = str.substring(start, i + 1);
        }
    }

    return { substring: longestSub, length: maxLength };
}

// Example usage
const input = 'abrkaabcdefghijjxxx';
const result = findLongestSubstring(input);
console.log(result);
