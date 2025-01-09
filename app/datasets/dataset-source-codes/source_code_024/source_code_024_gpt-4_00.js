// Function to strip HTML tags without using browser-based APIs
function stripHTMLTags(inputString) {
    // Create a regular expression that matches all HTML tags
    const regex = /(<([^>]+)>)/ig;
    
    // Use the replace method with our regex to replace all HTML tags with ''
    // thereby stripping them from the input string and returning the text content
    return inputString.replace(regex, '');
}

// Now, we would call our pure function with the given string example
let result = stripHTMLTags('<p>Hello <em>World</em>!</p>');
// Expected output: 'Hello World!'
