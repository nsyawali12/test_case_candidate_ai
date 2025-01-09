// Node.js script for longest substring without repeating characters

function findLongestSubstring(str) {
    let longestSubstring = '';
    let tempSubstring = '';
  
    for(let i = 0; i < str.length; i++) {
        const currChar = str[i];

        const indexOfChar = tempSubstring.indexOf(currChar);
        if(indexOfChar != -1) {
            if(tempSubstring.length > longestSubstring.length) {
                longestSubstring = tempSubstring;
            }

            // start tempSubstring from the next character
            tempSubstring = tempSubstring.slice(indexOfChar + 1) + currChar;
        } else {
            tempSubstring += currChar;
        }
    }
    
    if(tempSubstring.length > longestSubstring.length) {
        longestSubstring = tempSubstring;
    }
  
    // return the longest substring and its length
    return { substring: longestSubstring, length: longestSubstring.length };
}

console.log(findLongestSubstring('abrkaabcdefghijjxxx'));
