function removeDuplicates(str) {
    let result = ''; // Initialize result string without duplicates
    let seen = new Set(); // Create an empty set to track seen characters

    // Loop through each character in the string
    for (let char of str) {
        // If the character has not been seen, append it to the result
        if (!seen.has(char)) {
            result += char;
            seen.add(char); // Add character to the set
        }
    }
    return result; // Return the string with duplicates removed
}
