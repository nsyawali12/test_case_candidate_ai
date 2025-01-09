// Import the readline module to interact with the user via the console
const readline = require('readline');

// Create readline interface using process.stdin and process.stdout
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Ask the user for input
rl.question('Enter a string to find the longest substring without repeating characters: ', (input) => {
    console.log(findLongestSubstring(input)); // Find the longest substring and print the result
    rl.close(); // Close the readline interface
});

// Function to find the longest substring without repeating characters
function findLongestSubstring(str) {
    let start = 0, maxLength = 0, startIndex = 0;
    const charIndexMap = new Map();

    // Loop through each character in the string
    for (let i = 0; i < str.length; i++) {
        const char = str[i];

        // If the character was seen and is in the current window
        if (charIndexMap.has(char) && charIndexMap.get(char) >= start) {
            start = charIndexMap.get(char) + 1; // Move the start to the position after the last occurrence
        }
        
        charIndexMap.set(char, i); // Update last seen index of character
        
        // Check if we found a longer substring
        if (i - start + 1 > maxLength) {
            maxLength = i - start + 1;
            startIndex = start;
        }
    }
    
    // Return the longest substring without repeating characters
    return { substring: str.substring(startIndex, startIndex + maxLength), length: maxLength };
}
