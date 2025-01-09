function isPalindrome(s) {
  // Remove punctuation, convert to lowercase and remove whitespaces
  var cleanStr = s.replace(/[^A-Za-Z0-9]/g, '').toLowerCase();

  // Reverse the string
  var reverseStr = cleanStr.split('').reverse().join('');

  // Compare the string with its reversed version
  return cleanStr === reverseStr;
}
