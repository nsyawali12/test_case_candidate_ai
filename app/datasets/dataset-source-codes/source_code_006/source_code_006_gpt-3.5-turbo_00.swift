func longestIncreasingSubarray(_ nums: [Int]) -> [Int] {
    var start = 0
    var maxLength = 0
    var currentLength = 1
    
    for i in 1..<nums.count {
        if nums[i] > nums[i - 1] {
            currentLength += 1
        } else {
            // Reset currentLength when decreasing or equal
            currentLength = 1
        }
        
        if currentLength > maxLength {
            maxLength = currentLength
            start = i - currentLength + 1
        }
    }
    
    return Array(nums[start..<start+maxLength])
}

let input = [1, 2, 3, 2, 1]
let result = longestIncreasingSubarray(input)
print(result)  // Output: [1, 2, 3]
