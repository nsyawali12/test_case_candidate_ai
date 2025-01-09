function isAnagram(str1, str2) {
    // Function to prepare strings by removing non-alphabet characters, converting to lowercase, and sorting the letters
    const prepareString = (str) => {
        return str.toLowerCase().replace(/[^a-z]/g, "").split("").sort().join("");
    };

    // Prepare both strings
    const normalizedStr1 = prepareString(str1);
    const normalizedStr2 = prepareString(str2);

    // Compare the prepared strings
    return normalizedStr1 === normalizedStr2;
}
