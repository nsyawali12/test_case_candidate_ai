function anagramChecker(str1, str2) {
    // 1. Remove any non-alphabet character, and convert to lower case
    let formattedStr1 = str1.replace(/[^A-Za-z]/g, '').toLowerCase();
    let formattedStr2 = str2.replace(/[^A-Za-z]/g, '').toLowerCase();

    // 2. If lengths are not equal, they cannot be anagrams
    if (formattedStr1.length !== formattedStr2.length) {
        return false;
    }

    // 3. Convert each string into a sorted array
    let str1Arr = formattedStr1.split('').sort();
    let str2Arr = formattedStr2.split('').sort();

    // 4. Check if sorted arrays are equal
    for (let i = 0; i < str1Arr.length; i++) {
        if (str1Arr[i] !== str2Arr[i]) {
            return false;
        }
    }

    // 5. If all previous checks passed, the strings are anagrams
    return true;
}
