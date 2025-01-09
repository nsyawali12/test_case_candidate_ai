func findLongestSubArry(array: [Int]) -> [Int] {
        var finalStart = 0
        var finalCount = 0
        for i in 0..<array.count - 1{
            var start = 0
            var count = 0
            let index = i+1
            for j in index..<array.count {
                
                if(array[i] < array[j]) {
                    start = i
                    count = count + 1
                }
            }
            if(finalCount < count) {
                finalStart = start
                finalCount = count
            }
        }
        if(finalCount > 0) {
            var sub: [Int] = [Int]()
            let end = finalStart + finalCount
            for k in finalStart..<end {
                sub.append(array[k])
            }
            return sub
        }
        return []
    }