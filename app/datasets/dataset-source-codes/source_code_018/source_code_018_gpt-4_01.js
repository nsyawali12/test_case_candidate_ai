function isBalanced(str) {
    let depth = 0;
  
    for (let i = 0; i < str.length; i++) {
        if (str[i] === '(') {
            depth++;
        } else if (str[i] === ')') {
            if (depth === 0) return false;
            depth--;
        }
    }
  
    return depth === 0;
}
