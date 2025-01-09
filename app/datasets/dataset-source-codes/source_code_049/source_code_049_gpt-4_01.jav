import java.util.*;

public class Main {
    public static void main(String[] args) {
        System.out.println(isPalindrome("A man, a plan, a canal: Panama")); // returns: true
        System.out.println(isPalindrome("race a car")); // returns: false
    }

    public static boolean isPalindrome(String s) {
        // Initialize left and right pointers
        int left = 0, right = s.length() - 1;
        
        // Convert the string to lower case
        s = s.toLowerCase(Locale.ROOT);

        // Iterate over the characters in the string
        while (left < right) {
            // Move left pointer to next alphanumeric character
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }

            // Move right pointer to next alphanumeric character
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }

            // Compare the characters at the left and right pointers
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }

            // Move the pointers towards the center
            left++;
            right--;
        }

        // All characters matched, so the string is a palindrome
        return true;
    }
}
