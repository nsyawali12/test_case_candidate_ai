public class LargestElementFinder {
    public static void main(String[] args) {
        int[] array = {1, 4, 2, 9, 5};
        int largest = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > largest) {
                largest = array[i];
            }
        }
        
        System.out.println("Input: " + java.util.Arrays.toString(array));
        System.out.println("Output: " + largest);
    }
}
