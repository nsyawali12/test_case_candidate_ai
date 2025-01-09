import java.util.HashMap;
import java.util.Map;

public class TwoSum {
    public static int[] findTwoSum(int[] nums, int target) {
        Map<Integer, Integer> numIndices = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numIndices.containsKey(complement)) {
                return new int[] { numIndices.get(complement), i };
            }
            numIndices.put(nums[i], i);
        }
        return new int[] {}; // return an empty array if no solution
    }
    
    public static void main(String[] args) {
        int[] nums = {2, 7, 11, 15};
        int target = 9;
        int[] indices = findTwoSum(nums, target);
        if (indices.length == 2) {
            System.out.println("Output: [" + indices[0] + ", " + indices[1] + "]");
        } else {
            System.out.println("No solution found.");
        }
    }
}