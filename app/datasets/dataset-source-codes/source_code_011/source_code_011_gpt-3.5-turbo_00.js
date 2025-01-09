function removeDuplicates(str) {
    return [...new Set(str)].join('');
}

// Test the function with the example
console.log(removeDuplicates("hello")); // Output: "helo"
