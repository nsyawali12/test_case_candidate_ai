func longestIncreasingSubarray(nums: [Int]) -> [Int] {
    var current: [Int] = []
    var longest: [Int] = []

    // Check all elements in the array
    for num in nums {
        if current.isEmpty || num > current.last! {
            // Add current number to the list if it's greater than the last one
            current.append(num)
            if current.count > longest.count {
                // Update the longest sequence if the current one is longer
                longest = current
            }
        } else {
            // Start a new sequence if the current number is not greater
            current = [num]
        }
    }

    return longest
}

let input = [1, 2, 3, 2, 1]
let result = longestIncreasingSubarray(nums: input)
print(result) // Outputs: [1, 2, 3]
