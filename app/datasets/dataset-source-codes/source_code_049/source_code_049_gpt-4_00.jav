import java.util.*;

public class Main {
    public static void main(String[] args) {
        System.out.println(isPalindrome("A man, a plan, a canal: Panama")); // Output: true
        System.out.println(isPalindrome("race a car")); // Output: false
    }

    public static boolean isPalindrome(String s) {
        // convert the string to lower case and remove any non-alphanumeric characters
        String cleanedString = s
            .toLowerCase()
            .replaceAll("[^a-zA-Z0-9]", "");

        StringBuilder cleanedStringBuilder = new StringBuilder(cleanedString);
        
        // reverse the cleaned string
        String reversedString = cleanedStringBuilder.reverse().toString();
        
        // return if the cleaned string and the reversed string are equal
        return cleanedString.equals(reversedString);
    }
}
