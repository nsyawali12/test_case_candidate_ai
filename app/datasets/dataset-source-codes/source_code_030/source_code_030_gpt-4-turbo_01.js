function mergeSortedArrays(arr1, arr2) {
    let mergedArray = []; // Array to store the merged elements
    let i = 0, j = 0;

    // Iterate through both arrays and add the smaller of elements at current positions to the merged array
    while (i < arr1.length && j < arr2.length) {
        if (arr1[i] < arr2[j]) {
            mergedArray.push(arr1[i]);
            i++;
        } else {
            mergedArray.push(arr2[j]);
            j++;
        }
    }

    // Append remaining elements (if any) of arr1 to merged array
    while (i < arr1.length) {
        mergedArray.push(arr1[i]);
        i++;
    }

    // Append remaining elements (if any) of arr2 to merged array
    while (j < arr2.length) {
        mergedArray.push(arr2[j]);
        j++;
    }

    return mergedArray; // Return the merged array, now fully sorted
}
