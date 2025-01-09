// Java program to Reverse a String  
public class ReverseString { 
    public static void main(String[] args) { 
        String str = "hello"; 
        char chars[] = str.toCharArray(); // Convert string to character array
        for(int i = chars.length-1; i >= 0; i--) // Start from the end of the string
            System.out.print(chars[i]); // Print out the characters in reverse order
    } 
} 
