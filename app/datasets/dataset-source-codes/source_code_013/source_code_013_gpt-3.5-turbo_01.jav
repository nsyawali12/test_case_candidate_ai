public class StringReverse {

    public static void main(String[] args) {
        String input = "hello";
        String reversed = reverseString(input);
        System.out.println(reversed);
    }

    public static String reverseString(String str) {
        char[] charArray = str.toCharArray();
        int start = 0;
        int end = str.length() - 1;

        while (start < end) {
            // Swap characters at start and end indices
            char temp = charArray[start];
            charArray[start] = charArray[end];
            charArray[end] = temp;

            start++;
            end--;
        }

        return new String(charArray);
    }
}
