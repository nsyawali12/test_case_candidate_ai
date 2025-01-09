function bubbleSort(arr) {
    let len = arr.length;
    for (let i = len-1; i >=0; i--){
      for(let j = 1; j<=i; j++){
        // Swap numbers if previous number is greater than current one
        if(arr[j-1]>arr[j]){
            let temp = arr[j-1];
            arr[j-1] = arr[j];
            arr[j] = temp;
        }
      }
    }
    return arr;
}

// Testing the function
let arr = [34, 17, 23, 67, 45];
console.log(bubbleSort(arr)); // Expected Output: [17, 23, 34, 45, 67]
