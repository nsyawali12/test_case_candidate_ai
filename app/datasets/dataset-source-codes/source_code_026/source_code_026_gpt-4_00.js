import React, { useReducer, createContext, useState, useEffect } from 'react';

// Global state.
const initialState = {};
const Store = createContext(initialState);
const { Provider } = Store;

// Reducer function to handle user actions.
const reducer = (state, action) => {
    switch(action.type) {
        case 'ACTION_TYPE':
            return {
                ...state, 
                property: action.payload,
            };
        default:
            throw new Error();
    };
};

// Wrapping component with Provider to provide global state.
function StoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  return <Provider value={{ state, dispatch }}>{children}</Provider>;
}

// Use in a child component.
function SomeChildComponent() {
  const global = React.useContext(Store);
  const [localState, setLocalState] = useState();

  useEffect(() => {
    const data = getSomeData(global.state); // Assuming getSomeData is a utility function written to fetch data according to the global state
    setLocalState(data);
  }, [global]);

  return <div>{localState}</div>;
}

function ParentComponent() {
  return (
    <StoreProvider>
      <SomeChildComponent/>
    </StoreProvider>
  )
}

export default ParentComponent;
