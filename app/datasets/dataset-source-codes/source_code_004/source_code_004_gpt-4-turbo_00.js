function isPalindrome(s) {
    // Remove non-alphanumeric characters and convert to lower case
    const cleaned = s.replace(/[^A-Za-z0-9]/g, '').toLowerCase();

    // Initialize pointers for two ends
    let left = 0;
    let right = cleaned.length - 1;

    // Compare characters from start to end, moving towards the center
    while (left < right) {
        if (cleaned[left] !== cleaned[right]) {
            return false;
        }
        left++;
        right--;
    }

    return true;
}
