public class Main {
  public static void main(String[] args) {
    // Define the array.
    int[] array = {1, 4, 2, 9, 5};

    // Assume the first element is the largest initially.
    int largest = array[0];
    
    // Traverse array elements from the second one, and compare every element with the current largest.
    for (int i = 1; i < array.length; i++)
      if (array[i] > largest)
        largest = array[i];

    // Print the largest.
    System.out.println(largest);
  }
}
