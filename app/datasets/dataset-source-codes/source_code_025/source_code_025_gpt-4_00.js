function isAnagram(str1, str2) {
    // Remove spaces and convert to lower case
    let cleanedStr1 = str1.replace(/\s/g, '').toLowerCase();
    let cleanedStr2 = str2.replace(/\s/g, '').toLowerCase();

    // Check if lengths are the same
    if (cleanedStr1.length !== cleanedStr2.length) {
        return false;
    }

    // Create frequency counter objects
    let counter1 = {};
    let counter2 = {};

    // Fill the frequency counter objects
    for(let i = 0; i < cleanedStr1.length; i++) {
        if (/[a-zA-Z]/.test(cleanedStr1[i])) {
            counter1[cleanedStr1[i]] = (counter1[cleanedStr1[i]] || 0) + 1;
        }
        if (/[a-zA-Z]/.test(cleanedStr2[i])) {
            counter2[cleanedStr2[i]] = (counter2[cleanedStr2[i]] || 0) + 1;
        }
    }

    // Check if frequency counters are identical
    for(let key in counter1) {
        if (counter1[key] !== counter2[key]) {
            return false;
        }
    }

    return true;
}
