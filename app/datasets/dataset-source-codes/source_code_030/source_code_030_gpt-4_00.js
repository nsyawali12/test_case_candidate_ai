function mergeSortedArrays(array1, array2) {
  // Create a new array to store merged elements
  let mergedArray = [];

  // Indices to keep track of current element in each array
  let i = 0;
  let j = 0;

  // Loop until we've processed all elements in both arrays
  while (i < array1.length || j < array2.length) {
    // If we've processed all elements in array1, add remaining elements in array2
    // Or add element from array1 if it's smaller than current element in array2
    if (i === array1.length || (j < array2.length && array1[i] > array2[j])) {
      mergedArray.push(array2[j]);
      j++;
    } else {
      mergedArray.push(array1[i]);
      i++;
    }
  }

  return mergedArray;
}
