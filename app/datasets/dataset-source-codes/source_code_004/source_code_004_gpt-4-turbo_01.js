function isPalindrome(s) {
    // Normalize the string by removing non-alphanumeric characters and converting to lowercase
    const cleanString = s.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();

    // Use two pointers to compare characters from the start and the end of the string
    let left = 0;
    let right = cleanString.length - 1;

    while (left < right) {
        if (cleanString[left] !== cleanString[right]) {
            return false; // Characters do not match, it's not a palindrome
        }
        left++; // Move the left pointer to the right
        right--; // Move the right pointer to the left
    }

    return true; // All characters matched, it is a palindrome
}
