function mergeSortedArrays(array1, array2){
  // If one of arrays is empty, return the other one
  if(array1.length === 0) return array2;
  if(array2.length === 0) return array1;

  const mergedArray = [];
  let array1Item = array1[0]; // first element of first array
  let array2Item = array2[0]; // first element of second array
  let i = 1;  // to keep track of index in array1
  let j = 1;  // to keep track of index in array2

  // iterate over arrays
  while (array1Item || array2Item){
   // We push from array1 if array2 is empty OR if current element in array1 is <= current element in array2
   if(!array2Item || array1Item <= array2Item){
     mergedArray.push(array1Item);
     array1Item = array1[i]; // update to next element
     i++;
   } 
   // We push from array2 if array1 is empty OR current element in array2 < current element in array1
   else 
   {
     mergedArray.push(array2Item);
     array2Item = array2[j]; // update to next element
     j++;
   }
  }
  return mergedArray;
}
