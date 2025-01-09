function findMaxNumber(numbers) {
  // Find the maximum number in the array of numbers.
  return Math.max(...numbers);
}

// Example usage within a React component
function MaxNumberComponent({ numbers }) {
  return (
    <div>
      Maximum Number: {findMaxNumber(numbers)}
    </div>
  );
}

export default MaxNumberComponent;
