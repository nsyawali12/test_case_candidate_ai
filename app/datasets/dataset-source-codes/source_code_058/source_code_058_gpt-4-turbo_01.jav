public int[] twoSum(int[] nums, int target) {
    // Create a hashmap to store numbers and their indices
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        // Check if the complement exists in map
        if (map.containsKey(complement)) {
            // If complement exists, return the current index and the index of the complement
            return new int[] { map.get(complement), i };
        }
        // If complement doesn't exist, put the current number and its index into the map
        map.put(nums[i], i);
    }
    // Return an empty array if no solution is found
    return new int[] {};
}
