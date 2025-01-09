public class Main {
    public static void main(String[] args) {
        int[] array = {1, 4, 2, 9, 5}; // Example input array
        int largest = findLargestElement(array); // Finding the largest element
        System.out.println(largest); // Output the largest element
    }

    // Method to find the largest element in the array
    public static int findLargestElement(int[] arr) {
        if (arr == null || arr.length == 0) {
            throw new IllegalArgumentException("Array should not be empty");
        }
        int max = arr[0]; // Assume the first element is the largest initially

        // Traverse the array to find the largest element
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i]; // Update max if current element is greater
            }
        }
        return max; // Return the largest element
    }
}
