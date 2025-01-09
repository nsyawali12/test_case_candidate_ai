fun maxSumOfAnyContiguosSubarray(lists: List<List<Int>>) {
  lists.forEach { list ->
    var max = Int.MIN_VALUE
    var sum = 0
    list.forEach { value ->
      sum += value
      max = maxOf(sum, max)
      sum = maxOf(sum, 0)
    }
    result.add(max)
  }
}