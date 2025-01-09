// Node.js Function
function validateParentheses(str) {
    const stack = [];
    const pairs = {
        '(': ')',
    };

    for (let char of str) {
        if (char === '(') {
            stack.push(char);
        } else if (char === ')') {
            if (stack.length === 0 || pairs[stack.pop()] !== char) {
                return false;
            }
        }
    }

    return stack.length === 0;
}

// React Component
import React, { useState } from 'react';

const ParenthesesValidator = () => {
    const [input, setInput] = useState('');
  
    const handleInputChange = (e) => {
        setInput(e.target.value);
    };

    const isValid = validateParentheses(input);

    return (
        <div>
            <input
                type="text"
                value={input}
                onChange={handleInputChange}
            />
            {isValid ? "Output: true" : "Output: false"}
        </div>
    );
};

export default ParenthesesValidator;
