// Node.js function to check if the parentheses are balanced
function isBalanced(sequence) {
  let balance = 0;
  for (const char of sequence) {
    if (char === '(') {
      balance += 1;
    } else if (char === ')') {
      balance -= 1;
    }
    if (balance < 0) {
      return false;
    }
  }
  return balance === 0;
}

// React component to input sequence of parentheses and display the result
import React, { useState } from 'react';

function ParenthesesValidator() {
  const [input, setInput] = useState('');

  const handleChange = (event) => {
    setInput(event.target.value);
  };

  // Toggle the result based on current input using the isBalanced function
  // It updates as the user types, showing whether the current sequence is balanced.
  const result = isBalanced(input);

  return (
    <div>
      <input type="text" value={input} onChange={handleChange} placeholder="Enter parentheses" />
      <div>
        {input ? (result ? 'Balanced' : 'Not balanced') : 'Enter a sequence to validate'}
      </div>
    </div>
  );
}

export default ParenthesesValidator;
