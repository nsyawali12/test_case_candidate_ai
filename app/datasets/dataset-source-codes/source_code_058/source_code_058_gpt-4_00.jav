public class Solution {
    public int[] twoSum(int[] nums, int target) {
        // Initialize a new HashMap
        Map<Integer, Integer> num_map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            // If the map contains the complement, return the indices
            if (num_map.containsKey(complement)) {
                return new int[] { num_map.get(complement), i };
            }
            num_map.put(nums[i], i);
        }
        // If no solution, return an empty array
        return new int[]{};
    }
}
