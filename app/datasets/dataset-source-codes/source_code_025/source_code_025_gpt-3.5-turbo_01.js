function isAnagram(str1, str2) {
    const cleanString = (str) => str.replace(/[^a-zA-Z]/g, '').toLowerCase();
  
    const sortedStr1 = cleanString(str1).split('').sort().join('');
    const sortedStr2 = cleanString(str2).split('').sort().join('');

    return sortedStr1 === sortedStr2;
}

// Test cases
console.log(isAnagram('listen', 'silent')); // Output: true
console.log(isAnagram('hello', 'billion')); // Output: false
