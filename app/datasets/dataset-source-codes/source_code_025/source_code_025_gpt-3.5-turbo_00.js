function isAnagram(str1, str2) {
    const formatString = (str) => str.replace(/[^a-z]/gi, '').toLowerCase();
    const sortedStr1 = formatString(str1).split('').sort().join('');
    const sortedStr2 = formatString(str2).split('').sort().join('');
    
    return sortedStr1 === sortedStr2;
}

// Test cases
console.log(isAnagram('listen', 'silent')); // Output: true
console.log(isAnagram('hello', 'billion')); // Output: false
