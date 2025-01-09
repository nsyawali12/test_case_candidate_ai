public class ReverseWords {

    public static String reverseWords(String s) {
        // Split the input string on one or more spaces
        String[] words = s.split("\\s+");
        // StringBuilder to hold the reversed words
        StringBuilder reversed = new StringBuilder();
        
        // Iterate over the words array from the last element to the first
        for (int i = words.length - 1; i >= 0; i--) {
            reversed.append(words[i]); // Append the word to the StringBuilder
            if (i > 0) {
                reversed.append(" "); // Append a space if it's not the last word
            }
        }
        return reversed.toString(); // Convert StringBuilder to String and return it
    }

    // Main method to test the reverseWords function
    public static void main(String[] args) {
        String input = "Hello World!";
        String reversed = reverseWords(input);
        System.out.println(reversed);  // Output: "World! Hello"
    }
}
