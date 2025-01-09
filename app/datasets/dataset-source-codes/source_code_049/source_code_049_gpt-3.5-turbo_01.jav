public class PalindromeChecker {

    public static boolean isPalindrome(String str) {
        // Remove spaces, punctuation, and convert to lowercase
        String formattedStr = str.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        
        int left = 0;
        int right = formattedStr.length() - 1;
        
        while (left < right) {
            if (formattedStr.charAt(left) != formattedStr.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        
        return true;
    }

    public static void main(String[] args) {
        String input1 = "A man, a plan, a canal: Panama";
        String input2 = "race a car";
        
        System.out.println("Input: '" + input1 + "' Output: " + isPalindrome(input1));
        System.out.println("Input: '" + input2 + "' Output: " + isPalindrome(input2));
    }
}
