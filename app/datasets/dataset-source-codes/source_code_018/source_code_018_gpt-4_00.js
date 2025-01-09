// Node.js function to check if the parenthesis sequence is balanced or not
function isBalanced(parens) {
  let stack = [];

  for (let i = 0; i < parens.length; i++) {
    if (parens[i] === "(") {
      stack.push("(");
    } else if (parens[i] === ")") {
      if (stack.length === 0) {
        return false;
      }
      stack.pop();
    }
  }

  return stack.length === 0;
}
