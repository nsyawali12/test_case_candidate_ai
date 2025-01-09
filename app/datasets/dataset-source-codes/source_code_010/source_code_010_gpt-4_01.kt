fun maxSubArraySum(arrays: List<List<Int>>): List<Int> {
    val result = mutableListOf<Int>()
    
    // Iterate through each array in the list
    for (array in arrays) {
        var currentSum = array[0]
        var maxSum = array[0]
        
        // Compare each element with the previous sum add to current sum if it's greater
        for (i in 1 until array.size) {
            currentSum = Math.max(array[i], currentSum + array[i])
            maxSum = Math.max(maxSum, currentSum)
        }
        
        // Append the maximum sum of each sub array to the result list
        result.add(maxSum)
    }
    
    return result
}
