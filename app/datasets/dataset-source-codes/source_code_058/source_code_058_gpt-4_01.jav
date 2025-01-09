public int[] twoSum(int[] nums, int target) {
    // Create a map to hold the numbers and their indices
    Map<Integer, Integer> numIndices = new HashMap<>();
    
    // Iterate over the array
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        // If the map contains the complement, return its index and the current index
        if (numIndices.containsKey(complement)) {
            return new int[] { numIndices.get(complement), i };
        }
        // Otherwise, store this number and its index in the map
        numIndices.put(nums[i], i);
    }

    // If no solution, return an empty array
    return new int[] {};
}
