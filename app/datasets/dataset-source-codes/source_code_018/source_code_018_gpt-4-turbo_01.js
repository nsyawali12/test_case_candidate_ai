// Node.js function to check if parentheses are balanced
function isBalancedParens(sequence) {
    let balance = 0;
    for (let i = 0; i < sequence.length; i++) {
        if (sequence[i] === '(') balance++;
        else if (sequence[i] === ')') balance--;
        if (balance < 0) return false; // Early exit if more closing than opening
    }
    return balance === 0;
}

// React component that uses the isBalancedParens function
import React, { useState } from 'react';

function ParenthesesValidator() {
    const [input, setInput] = useState('');
    const [isValid, setIsValid] = useState(true);

    const handleChange = (event) => {
        const value = event.target.value;
        setInput(value);
        setIsValid(isBalancedParens(value));
    };

    return (
        <div>
            <input
                type="text"
                value={input}
                onChange={handleChange}
                placeholder="Enter parentheses sequence"
            />
            <div>
                {isValid ? 'Sequence is balanced' : 'Sequence is not balanced'}
            </div>
        </div>
    );
}

export default ParenthesesValidator;
