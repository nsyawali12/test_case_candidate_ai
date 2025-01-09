public class Solution {
    public int[] twoSum(int[] nums, int target) {
        // HashMap to store the indices of the numbers
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            // Check if the complement exists in the map
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                // If complement exists, return the pair of indices
                return new int[] {map.get(complement), i};
            }
            // Store the index of the current number in the map
            map.put(nums[i], i);
        }
        // Return an empty array if no solution is found
        return new int[] {};
    }
}
