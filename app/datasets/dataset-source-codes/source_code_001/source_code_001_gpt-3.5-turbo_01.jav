public class ReverseWords {
    public static String reverseWordsInString(String input) {
        if (input == null || input.isEmpty()) {
            return "";
        }
        
        String[] words = input.split(" ");
        StringBuilder reversed = new StringBuilder();
        
        for (int i = words.length - 1; i >= 0; i--) {
            reversed.append(words[i]);
            
            if (i != 0) {
                reversed.append(" ");
            }
        }
        
        return reversed.toString();
    }

    public static void main(String[] args) {
        String input = "Hello World!";
        String reversedWords = reverseWordsInString(input);
        System.out.println(reversedWords); // Output: "World! Hello"
    }
}
