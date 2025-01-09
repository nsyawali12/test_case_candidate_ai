// Declare a function that takes a string as a parameter
function stripHTMLTags(str) {
    // Use the replace method with a regular expression to identify HTML tags
    // and replace them with an empty string
    let newStr = str.replace(/(<([^>]+)>)/gi, "");
    // Return the resulting string, which should be the original string without HTML tags
    return newStr;
}
