fun maxSubArray(nums: List<Int>): Int {
    var maxSum = nums[0]
    var currSum = nums[0]
  
    for (i in 1 until nums.size) {
        // If the current sum becomes negative, start a new subarray.
        currSum = Math.max(nums[i], currSum + nums[i])
        // Always maintain the maximum sum seen so far.
        maxSum = Math.max(maxSum, currSum)
    }
    return maxSum
}

fun main() {
    val input = listOf(
        listOf(-2, 1, -3, 4, -1, 2, 1, -5, 4),
        listOf(1, 2, 3, 4, 5),
        listOf(5, 4, 3, 2, 1),
        listOf(-1, -2, -3, -4, -5)
    )
  
    input.forEach { println(maxSubArray(it)) }
}
