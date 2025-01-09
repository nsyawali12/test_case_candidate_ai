function removeDuplicates(str) {
    // Create a Set to keep track of unique characters
    const uniqueChars = new Set();
    
    // Iterate through each character in the string
    for (const char of str) {
        uniqueChars.add(char); // Add the character to the Set, duplicates will automatically be ignored
    }
    
    // Join the unique characters into a string and return it
    return Array.from(uniqueChars).join('');
}
