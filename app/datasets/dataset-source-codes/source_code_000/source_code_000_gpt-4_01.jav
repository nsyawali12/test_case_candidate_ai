public class Main {
    public static void main(String[] args) {
        // Define the input array
        int[] array = {1, 4, 2, 9, 5};

        // Initialize the largestElement to the first element in the array
        int largestElement = array[0];

        // Iterate over the array
        for (int i = 1; i < array.length; i++){
            // Check if array[i] is greater than largestElement
            if (array[i] > largestElement) {
                // If yes, assign array[i] to largestElement
                largestElement = array[i];
            }
        }

        // Print the largest element in the array
        System.out.println(largestElement);
    }
}
