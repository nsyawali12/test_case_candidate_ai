func longestIncreasingSubarray(arr: [Int]) -> [Int] {
    if arr.isEmpty { return [] }
    
    var maxLength = 1 // Length of the longest increasing subarray
    var currentLength = 1 // Length of the current increasing subarray
    var startIndex = 0 // Start index of the best (longest so far) increasing subarray

    for i in 1..<arr.count {
        // Check if the current element is greater than the previous one
        if arr[i] > arr[i - 1] {
            currentLength += 1
        } else {
            // Check if the current increasing subarray is the longest
            if currentLength > maxLength {
                maxLength = currentLength
                startIndex = i - currentLength
            }
            // Reset the length
            currentLength = 1
        }
    }

    // Update the longest subarray in case the longest was found at the end of the array
    if currentLength > maxLength {
        maxLength = currentLength
        startIndex = arr.count - currentLength
    }

    // Return the longest increasing subarray
    return Array(arr[startIndex..<(startIndex + maxLength)])
}
