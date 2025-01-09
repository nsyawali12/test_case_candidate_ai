import React from 'react';

const FindMaxNumber = ({ numbers }) => {
    if (numbers.length === 0) {
        return <p>No numbers provided</p>;
    }

    const maxNumber = Math.max(...numbers);

    return (
        <p>Maximum number in the array: {maxNumber}</p>
    );
};

// Example usage: <FindMaxNumber numbers={[1, 5, 9, 3, 7]} />

export default FindMaxNumber;
