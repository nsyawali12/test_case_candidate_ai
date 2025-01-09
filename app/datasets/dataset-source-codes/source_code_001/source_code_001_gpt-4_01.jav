public class Main {
    public static void main(String[] args) {
        System.out.println(reverseWords("Hello World!"));
    }

    // Function to reverse words
    public static String reverseWords(String str) {
        // Split input string separated by space
        String[] words = str.split(" ");
        String reversedString = "";

        // Reverse each word's position
        for (int i = words.length - 1; i >= 0; i--) {
            // Append each word to the string
            reversedString += words[i] + " ";
        }
        
        // Remove the trailing space and return the reversed string
        return reversedString.trim();
    }
}
