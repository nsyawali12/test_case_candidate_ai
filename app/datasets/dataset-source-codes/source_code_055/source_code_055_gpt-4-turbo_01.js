function bubbleSort(arr) {
    let n = arr.length;
    let swapped;
    do {
        swapped = false;
        for (let i = 1; i < n; i++) {
            // Compare the adjacent positions
            if (arr[i - 1] > arr[i]) {
                // Swap if elements in wrong order
                let temp = arr[i - 1];
                arr[i - 1] = arr[i];
                arr[i] = temp;
                swapped = true;
            }
        }
        // Reduce n since the last element is already in place
        n--;
    } while (swapped);
    return arr;
}

// Example usage:
// const sortedArray = bubbleSort([34, 17, 23, 67, 45]);
// console.log(sortedArray); // Output: [17, 23, 34, 45, 67]
