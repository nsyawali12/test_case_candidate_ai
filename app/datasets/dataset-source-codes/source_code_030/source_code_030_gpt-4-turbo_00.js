function mergeSortedArrays(arr1, arr2) {
    const mergedArray = [];
    let i = 0, j = 0;

    // Iterate over both arrays and insert the smallest value into mergedArray
    while (i < arr1.length && j < arr2.length) {
        if (arr1[i] < arr2[j]) {
            mergedArray.push(arr1[i]);
            i++;
        } else {
            mergedArray.push(arr2[j]);
            j++;
        }
    }

    // Append remaining elements from arr1 or arr2
    while (i < arr1.length) {
        mergedArray.push(arr1[i]);
        i++;
    }
    
    while (j < arr2.length) {
        mergedArray.push(arr2[j]);
        j++;
    }

    return mergedArray;
}
