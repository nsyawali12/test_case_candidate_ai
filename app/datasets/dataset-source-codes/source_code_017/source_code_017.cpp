uint8_t* getMiddleChars(uint8_t* input, uint32_t len) {
  // can we use new? do we have to import string? dont remember lib name.
  uint8_t* result = malloc() // dont remember exactly malloc syntax
  
  // if even
  if (len % 2 == 0) {
    // returns the char in the middle
    return memcpy(result, input[(len-1)/2], 1)
  } else {
    return memcpy(result, input[(len-1)/2], 2)
  } 
}