public class PalindromeChecker {

    public static boolean isPalindrome(String s) {
        s = s.toLowerCase().replaceAll("[^a-z0-9]", ""); // Normalize: remove non-alphanumeric and convert to lower case
        int left = 0; // Starting index
        int right = s.length() - 1; // Ending index

        while (left < right) { // Compare characters from each end
            if (s.charAt(left) != s.charAt(right)) {
                return false; // If characters do not match, it's not a palindrome
            }
            left++; // Move inward from the left side
            right--; // Move inward from the right side
        }

        return true; // If no mismatches, it is a palindrome
    }

    public static void main(String[] args) {
        System.out.println(isPalindrome("A man, a plan, a canal: Panama")); // Output: true
        System.out.println(isPalindrome("race a car")); // Output: false
    }
}
