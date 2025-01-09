function removeDuplicateChars(str) {
  let result = '';
  // loop through the string characters
  for(let i=0; i<str.length; i++) {
    const char = str.at(i);
    // if the character is not in result string, append it
    if(result.indexOf(char) === -1) {
      result += char;
    }
  }

  return result;
}