func longestIncreasingSubarray(_ arr: [Int]) -> [Int] {
    if arr.isEmpty {
        return []
    }
    var currentSequence = [arr[0]]
    var longestSequence = [arr[0]]
    for i in 1..<arr.count {
        if arr[i] > arr[i - 1] {
            currentSequence.append(arr[i])
            if currentSequence.count > longestSequence.count {
                longestSequence = currentSequence
            }
        } else {
            currentSequence = [arr[i]]
        }
    }
    return longestSequence
}
