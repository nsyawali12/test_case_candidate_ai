// Function to find the longest substring without repeating characters
function findLongestSubstring(input) {
    let longestSubstring = '';
    let currentSubstring = '';
    
    for (let char of input) {
        if (currentSubstring.includes(char)) {
            if (currentSubstring.length > longestSubstring.length) {
                longestSubstring = currentSubstring;
            }
            currentSubstring = currentSubstring.slice(currentSubstring.indexOf(char) + 1);
        }
        currentSubstring += char;
    }
    
    if (currentSubstring.length > longestSubstring.length) {
        longestSubstring = currentSubstring;
    }
    
    return {
        substring: longestSubstring,
        length: longestSubstring.length
    };
}

// Input string
const input = 'abrkaabcdefghijjxxx';
const result = findLongestSubstring(input);

console.log("Input:", input);
console.log("Output:", result);
