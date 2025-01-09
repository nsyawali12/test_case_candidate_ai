// Node.js function to check if a sequence of parentheses is balanced
function isBalanced(input) {
    let stack = [];
    
    for (let char of input) {
        if (char === '(') {
            stack.push(char);
        } else if (char === ')') {
            if (stack.length === 0) {
                return false; // Unbalanced if closing parenthesis without a matching opening parenthesis
            }
            stack.pop();
        }
    }
    
    return stack.length === 0; // Return true if stack is empty (all parentheses matched)
}

module.exports = {
    isBalanced
};
