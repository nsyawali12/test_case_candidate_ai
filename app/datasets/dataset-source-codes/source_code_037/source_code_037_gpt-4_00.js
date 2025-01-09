// Node.js script to find the longest substring without repeating characters
function longestSubstring(input){
    let start = 0; 
    let maxLen = 0; 
    let map = new Map();
    let extendedStr; 
  
    for(let i=0; i<input.length; i++){
        const ch = input.charAt(i);
  
        if(map.has(ch)){
            start = Math.max(start,map.get(ch)+1);
        }

        // We add current character to the map
        map.set(ch,i);
    
        // We update maximum length if necessary
        if(maxLen < i-start+1){
            maxLen = i-start+1;
            extendedStr = input.substring(start, i+1);
        }
    }
  
    // We return the longest substring without repeating characters and its length
    return { substring: extendedStr, length: maxLen };
}
  
// We execute the function with an example string
console.log(longestSubstring("abrkaabcdefghijjxxx"));
