func longestIncreasingSubarray(arr: [Int]) -> [Int] {
    // Guard against empty array input
    guard !arr.isEmpty else { return [] }

    var maxLength = 1
    var currentLength = 1
    
    // To store the starting position of the longest subarray found
    var maxStartIndex = 0
    var currentStartIndex = 0
    
    // Iterate over the array to find the longest increasing subarray
    for i in 1..<arr.count {
        if arr[i] > arr[i - 1] {
            currentLength += 1
        } else {
            if currentLength > maxLength {
                maxLength = currentLength
                maxStartIndex = currentStartIndex
            }
            // Reset current length and start index
            currentLength = 1
            currentStartIndex = i
        }
    }
    
    // Final check in case the longest subarray is at the end of the array
    if currentLength > maxLength {
        maxLength = currentLength
        maxStartIndex = currentStartIndex
    }
    
    // Extract the longest increasing subarray using the start index and length
    return Array(arr[maxStartIndex..<maxStartIndex + maxLength])
}
