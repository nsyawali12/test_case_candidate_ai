func longestIncreasingSubarray(_ nums: [Int]) -> [Int] {
    if nums.isEmpty {
        return []
    }
    
    var currentSubarray: [Int] = [nums[0]]
    var longestSubarray: [Int] = [nums[0]]
    
    for i in 1..<nums.count {
        if nums[i] > nums[i - 1] {
            currentSubarray.append(nums[i])
        } else {
            if currentSubarray.count > longestSubarray.count {
                longestSubarray = currentSubarray
            }
            currentSubarray = [nums[i]]
        }
    }
    
    if currentSubarray.count > longestSubarray.count {
        longestSubarray = currentSubarray
    }
    
    return longestSubarray
}

// Test the function with example input
let input = [1, 2, 3, 2, 1]
let result = longestIncreasingSubarray(input)
print(result)
