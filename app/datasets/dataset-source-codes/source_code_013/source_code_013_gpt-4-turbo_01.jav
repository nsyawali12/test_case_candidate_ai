public class StringReversal {
    public static void main(String[] args) {
        String input = "hello";
        String reversed = reverseString(input);
        System.out.println(reversed);
    }

    // Method to reverse a string without using built-in functions
    public static String reverseString(String str) {
        char[] characters = str.toCharArray();
        int left = 0;
        int right = characters.length - 1;

        // Swap characters to reverse the string
        while (left < right) {
            char temp = characters[left];
            characters[left] = characters[right];
            characters[right] = temp;
            left++;
            right--;
        }

        // Create the reversed string from characters array
        return new String(characters);
    }
}
