function removeDuplicates(str) {
    return Array.from(new Set(str)).join('');
}

// Example
console.log(removeDuplicates("hello")); // Output: "helo"
