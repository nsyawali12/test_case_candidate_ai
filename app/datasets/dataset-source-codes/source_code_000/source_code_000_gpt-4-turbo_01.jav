public class LargestElement {
    public static void main(String[] args) {
        int[] arr = {1, 4, 2, 9, 5}; // example input
        int maxElement = getMax(arr);
        System.out.println(maxElement); // output the largest number
    }

    // Function to find the largest element in an array
    public static int getMax(int[] array) {
        int max = array[0]; // Assume first element is the largest initially

        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) { // Check if current element is greater than max
                max = array[i];  // Update max if current element is larger
            }
        }
        return max; // Return the largest element
    }
}
