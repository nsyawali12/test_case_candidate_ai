function bubbleSort(arr) {
    let n = arr.length;
    let swapped;
    do {
        swapped = false;
        for (let i = 1; i < n; i++) {
            if (arr[i - 1] > arr[i]) {
                // Swap elements if they are in wrong order
                [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
                swapped = true;
            }
        }
        // Reduce n because the last element is already in place
        n--;
    } while (swapped);

    return arr;
}

// Example usage:
// console.log(bubbleSort([34, 17, 23, 67, 45]));  // Output: [17, 23, 34, 45, 67]
