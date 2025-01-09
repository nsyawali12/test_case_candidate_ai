public class PalindromeChecker {
    
    public static boolean isPalindrome(String s) {
        // Clean the string: remove non-alphanumeric characters and convert to lower case
        String cleaned = s.replaceAll("[^A-Za-z0-9]", "").toLowerCase();

        // Initialize left and right pointers
        int left = 0, right = cleaned.length() - 1;

        // Check if the string reads the same from the front and back
        while (left < right) {
            if (cleaned.charAt(left) != cleaned.charAt(right)) {
                return false; // Characters do not match
            }
            left++;
            right--;
        }

        return true; // String is a palindrome
    }

    public static void main(String[] args) {
        String input1 = "A man, a plan, a canal: Panama";
        System.out.println(isPalindrome(input1)); // Output: true

        String input2 = "race a car";
        System.out.println(isPalindrome(input2)); // Output: false
    }
}
