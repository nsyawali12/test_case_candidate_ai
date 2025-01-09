fun maxSubArraySum(nums: List<Int>): Int {
    var maxSoFar = nums[0]
    var currentMax = nums[0]

    for (i in 1 until nums.size) {
        currentMax = maxOf(nums[i], currentMax + nums[i]) // maximum of current element or current element + previous maximum
        maxSoFar = maxOf(maxSoFar, currentMax) // update the maximum found so far
    }

    return maxSoFar
}

fun main() {
    val testCases = listOf(
        listOf(-2, 1, -3, 4, -1, 2, 1, -5, 4),
        listOf(1, 2, 3, 4, 5),
        listOf(5, 4, 3, 2, 1),
        listOf(-1, -2, -3, -4, -5)
    )

    val results = testCases.map { maxSubArraySum(it) }
    println(results) // Output should be: [6, 15, 15, -1]
}
