public class ReverseString {
    public static void main(String[] args) {
        String input = "hello";  // The original string
        String reversed = "";   // This will store the reversed string

        // Looping from the end of the string to the beginning
        for (int i = input.length() - 1; i >= 0; i--) {
            reversed += input.charAt(i);  // Adding each character to the reversed string
        }

        System.out.println(reversed);  // Output the reversed string
    }
}
