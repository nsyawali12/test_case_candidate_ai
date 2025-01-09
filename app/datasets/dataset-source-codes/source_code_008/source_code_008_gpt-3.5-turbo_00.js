import React from 'react';

class MaxNumberFinder extends React.Component {
  findMaxNumber = (numbers) => {
    if (numbers.length === 0) {
      return null; // Handle case where array is empty
    }

    let maxNumber = numbers[0];
    for (let i = 1; i < numbers.length; i++) {
      if (numbers[i] > maxNumber) {
        maxNumber = numbers[i];
      }
    }

    return maxNumber;
  }

  render() {
    const numbers1 = [1, 5, 9, 3, 7];
    const numbers2 = [-2, 0, -5, -1];

    const maxNumber1 = this.findMaxNumber(numbers1);
    const maxNumber2 = this.findMaxNumber(numbers2);

    return (
      <div>
        <p>- input: {JSON.stringify(numbers1)}</p>
        <p>output: {maxNumber1}</p>
        <br />
        <p>- input: {JSON.stringify(numbers2)}</p>
        <p>output: {maxNumber2}</p>
      </div>
    );
  }
}
