function removeDuplicates(str) {
  let uniqueStr = ''; // Initialize the result string as empty
  for (let i = 0; i < str.length; i++) { // Iterate through each character of the string
    if (uniqueStr.indexOf(str[i]) == -1) { // Check if the character is already in the result string
      uniqueStr += str[i]; // If not, append it to the result string
    }
  }
  return uniqueStr; // Return the result string
}
//Call the function with the provided example
console.log(removeDuplicates('hello')); // Should print 'helo'
