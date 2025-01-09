fun findLargestElement(array: IntArray) : Int {
  if (array.isEmpty()) {
    return -1
  }

  var maximum = array[0]

  for (item in array) {
    if (item > maximum) {
      maximum = item
    }
  }

  return maximum
    
}