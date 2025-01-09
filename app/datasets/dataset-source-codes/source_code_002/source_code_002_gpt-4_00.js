function findOdd(arr) {
  let res = [];
  let xor2 = 0;

  // XOR of all elements
  for(let i=0; i<arr.length; i++)
    xor2 = xor2 ^ arr[i];

  // Rightmost set bit
  let set_bit_no = xor2 & ~(xor2 - 1);

  let x = 0, y = 0;
  
  /* Dividing elements in two sets: 
   1. The elements having the corresponding bit as 1. 
   2. The elements having the corresponding bit 0. */
  for(let i=0; i<arr.length; i++) {
    if((arr[i] & set_bit_no) != 0)
      x = x ^ arr[i]; // XOR of elements having set as rightmost bit
    else
      y = y ^ arr[i]; // XOR of elements having unset as rightmost bit
  }

  //push elements to results array
  res.push(x);
  res.push(y);
  
  return res;
}
