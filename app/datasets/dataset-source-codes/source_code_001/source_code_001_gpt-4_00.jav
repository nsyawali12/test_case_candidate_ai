public class Main {
    public static void main(String[] args) {
        String str = "Hello World!";
        System.out.println(reverseWords(str));
    }
 
    // Function to reverse words of a String
    static String reverseWords(String str)
    {
        // Split the input String into words
        String[] words = str.split(" ");
        
        // Initialize result
        String result = "";
        
        // Process each word
        for (int i = words.length-1; i >= 0; i--) {
            result = result + (i != words.length - 1 ? " " : "") + words[i];
        }
        
        // Return the result
        return result;
    }
}
