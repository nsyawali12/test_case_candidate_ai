function isAnagram(str1, str2) {
    // Normalize the strings: remove non-alphabet characters, convert to lowercase
    const normalize = str => str.toLowerCase().replace(/[^a-z]/g, '');
    str1 = normalize(str1);
    str2 = normalize(str2);
    
    // Check if lengths are different
    if (str1.length !== str2.length) {
        return false;
    }
    
    // Sort the letters of both strings
    const sortedStr1 = str1.split('').sort().join('');
    const sortedStr2 = str2.split('').sort().join('');
    
    // Compare sorted strings
    return sortedStr1 === sortedStr2;
}
