function findLongestSubstring(input) {
    let longestSubstring = '';
    let currentSubstring = '';
    let seenChars = new Set();
  
    for (let char of input) {
        if (!seenChars.has(char)) {
            currentSubstring += char;
            seenChars.add(char);
            if (currentSubstring.length > longestSubstring.length) {
                longestSubstring = currentSubstring;
            }
        } else {
            let index = currentSubstring.indexOf(char);
            currentSubstring = currentSubstring.slice(index + 1) + char;
            seenChars = new Set(currentSubstring);
        }
    }
  
    return { substring: longestSubstring, length: longestSubstring.length };
}
  
const input = 'abrkaabcdefghijjxxx';
const result = findLongestSubstring(input);
console.log(result);
