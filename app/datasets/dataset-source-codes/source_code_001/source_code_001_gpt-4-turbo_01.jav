public class ReverseWords {
    public static void main(String[] args) {
        String input = "Hello World!";
        String reversed = reverseWords(input);
        System.out.println(reversed);
    }

    // Function to reverse the order of words in a given string
    public static String reverseWords(String s) {
        String[] words = s.split(" "); // Split the string into words
        StringBuilder reversed = new StringBuilder();

        // Loop from the end to the beginning of the words array
        for(int i = words.length - 1; i >= 0; i--) {
            reversed.append(words[i]); // Append the words in reverse order
            if (i > 0) { // Append space if it's not the last word
                reversed.append(" ");
            }
        }
        return reversed.toString();
    }
}
