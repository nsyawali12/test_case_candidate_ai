fun maxSubarraySum(nums: IntArray): Int {
    var maxSum = Int.MIN_VALUE
    var currentSum = 0
    
    for (num in nums) {
        currentSum = maxOf(num, currentSum + num)
        maxSum = maxOf(maxSum, currentSum)
    }
    
    return maxSum
}

fun main() {
    val arrays = listOf(
        intArrayOf(-2, 1, -3, 4, -1, 2, 1, -5, 4),
        intArrayOf(1, 2, 3, 4, 5),
        intArrayOf(5, 4, 3, 2, 1),
        intArrayOf(-1, -2, -3, -4, -5)
    )

    val maxSums = arrays.map { maxSubarraySum(it) }
    println(maxSums) // Output: [6, 15, 15, -1]
}
