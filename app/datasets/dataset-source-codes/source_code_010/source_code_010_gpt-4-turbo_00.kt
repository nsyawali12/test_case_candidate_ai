fun maxSubArraySum(nums: IntArray): Int {
    var maxCurrent = nums[0]
    var maxGlobal = nums[0]

    for (i in 1 until nums.size) {
        maxCurrent = maxOf(nums[i], maxCurrent + nums[i])
        maxGlobal = maxOf(maxGlobal, maxCurrent)
    }

    return maxGlobal
}

// Usage example
fun main() {
    val arrays = listOf(
        intArrayOf(-2,1,-3,4,-1,2,1,-5,4),
        intArrayOf(1,2,3,4,5),
        intArrayOf(5,4,3,2,1),
        intArrayOf(-1,-2,-3,-4,-5)
    )

    // Mapping each array to its max subarray sum
    val results = arrays.map { maxSubArraySum(it) }
    println(results) // Expected Output: [6, 15, 15, -1]
}
