const isPalindrome = (str) => {
  str = str.toLowerCase().replace(/[\W_]/g, '');
  const reversed = str.split('').reverse().join('');
  return str === reversed;
}

// Test cases
console.log(isPalindrome("racecar")); // true
console.log(isPalindrome("hello")); // false
