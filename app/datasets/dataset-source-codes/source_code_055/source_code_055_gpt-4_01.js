function bubbleSort(array) {
  let length = array.length;
  let swapped;

  // Loop through the array
  for(let i = 0; i < length; i++) {
    swapped = false;

    // Loop through the array from 0 to length-i-1
    // Last i elements are already in place
    for(let j = 0; j < length-i-1; j++) {
      // Swap if the element found is greater than the next element
      if(array[j] > array[j+1]) {
        const temp = array[j];
        array[j] = array[j+1];
        array[j+1] = temp;
        swapped = true;
      }
    }

    // If no two elements were swapped in the inner loop, the array is sorted
    if(!swapped) {
      break;
    }
  }

  return array;
}

// To utilize the function:
const array = [34, 17, 23, 67, 45];
console.log(bubbleSort(array)); // This will output: [17, 23, 34, 45, 67]
