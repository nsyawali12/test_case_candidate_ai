// I have created a few components with an common scenario to prevent 
// children components from rerendering if the related props do not change
// even when the parent component got updated using the HOC that is 'memo'

// there will be different scenarios with different setup and complexity
// in such cases, other approaching such as using useCallback for memoizing the function
// useMemo for memoizing the result of the function

import { memo, useState } from 'react';

const App = () => {
  const [counter, setCounter] = useState(0);
  const [light, setLight] = useState(false);

  return (
    <div>
      <Counter counter={counter} />
      <button onClick={() => setCounter((c) => c - 1)}>Decrement</button>
      <button onClick={() => setCounter((c) => c + 1)}>Increment</button>
      <br /> <br /> <br />
      <Light light={light} />
      <button onClick={() => setLight((c) => !c)}>Toggle</button>
    </div>
  );
};

export default App;

const Counter = memo(({ counter }) => {
  console.log('counter', counter);
  return <div>Counter: {counter}</div>;
});

const Light = memo(({ light }) => {
  console.log('light', light);
  return <div>Light: {light ? 'On' : 'Off'}</div>;
});