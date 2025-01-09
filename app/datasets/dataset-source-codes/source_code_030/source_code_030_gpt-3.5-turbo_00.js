function mergeSortedArrays(arr1, arr2) {
    let sortedArray = [];
    let i = 0;
    let j = 0;

    while (i < arr1.length && j < arr2.length) {
        if (arr1[i] < arr2[j]) {
            sortedArray.push(arr1[i]);
            i++;
        } else {
            sortedArray.push(arr2[j]);
            j++;
        }
    }

    while (i < arr1.length) {
        sortedArray.push(arr1[i]);
        i++;
    }

    while (j < arr2.length) {
        sortedArray.push(arr2[j]);
        j++;
    }

    return sortedArray;
}

// Test the function with the example input
const arr1 = [0, 3, 4, 31];
const arr2 = [4, 6, 30];
console.log(mergeSortedArrays(arr1, arr2)); // Output: [0, 3, 4, 4, 6, 30, 31]
