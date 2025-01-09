function isPalindrome(str) {
  // Removing all punctuation, non-alphanumeric characters, and making the string lower case.
  let cleanedStr = str.replace(/[^0-9a-z]/gi, '').toLowerCase();

  // Reversing the string
  let reversedStr = cleanedStr.split('').reverse().join('');

  // Check if reversed string is the same as the cleaned string
  return cleanedStr === reversedStr;
}
