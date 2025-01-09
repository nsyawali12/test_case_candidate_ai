public class Main {
    public static void main(String[] args) {
        int[] arr = {1, 4, 2, 9, 5};
        int max = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }

        System.out.println("Input: [1, 4, 2, 9, 5]");
        System.out.println("Output: " + max);
    }
}
