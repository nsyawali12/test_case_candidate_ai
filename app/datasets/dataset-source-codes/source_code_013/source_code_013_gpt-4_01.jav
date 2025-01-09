public class Main {
    public static void main(String[] args) {
        String input = "hello";
        String reversedString = reverseString(input);
        System.out.println(reversedString);
    }

    // Define a method to reverse the string
    private static String reverseString(String str) {
        char[] characters = str.toCharArray(); // Convert the string to an array of characters
        int left = 0, right = str.length() - 1; // Define the left and right pointers
        while (left < right) {
            char temp = characters[left]; // Save the left character
            characters[left] = characters[right]; // Replace the left character by the right character
            characters[right] = temp; // Replace the right character by the saved left character
            left++; // Move the left pointer to the right
            right--; // Move the right pointer to the left
        }
        return new String(characters); // Convert the array of characters back to a string and return it
    }
}
