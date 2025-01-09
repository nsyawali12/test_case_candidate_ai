function findOddOccurrences(arr) {
    // Create an object to store occurrences
    var occurrences = {};
  
    // Iterate over each element in the array
    for(var i = 0; i < arr.length; i++) {
        // Check if the element is already in the object
        // Increase the count if it exists
        // If not, set the count to 1
        if(occurrences[arr[i]]) {
            occurrences[arr[i]]++;
        } else {
            occurrences[arr[i]] = 1;
        }
    }
  
    // Initialize an empty array to store elements with odd occurrences
    var result = [];
  
    // Iterate over the occurrences object
    for(var prop in occurrences) {
        // Check if the count of each element is odd
        // If it's odd, push the numerical value into result array
        if(occurrences[prop] % 2 !== 0) {
            result.push(parseInt(prop));
        }
    }
  
    // Return the result array
    return result;
}
