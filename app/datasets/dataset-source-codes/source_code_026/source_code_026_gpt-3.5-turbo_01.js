import React, { useState, useEffect, useMemo } from 'react';

const DataProvider = ({ data, children }) => {
  const [updatedData, setUpdatedData] = useState(data);

  useEffect(() => {
    setUpdatedData(data);
  }, [data]);

  return <>{children(updatedData)}</>;
};

const ComponentA = ({ data }) => {
  console.log('ComponentA rendered');
  
  return <div>{data}</div>;
};

const ComponentB = ({ data }) => {
  console.log('ComponentB rendered');
  
  return <div>{data}</div>;
};

const App = () => {
  const [value, setValue] = useState('');
  
  const handleInputChange = (e) => {
    setValue(e.target.value);
  };
  
  // Wrap components that depend on 'value' in DataProvider to optimize their rendering
  const optimizedComponentA = useMemo(() => <ComponentA data={value} />, [value]);
  const optimizedComponentB = useMemo(() => <ComponentB data={value} />, [value]);

  return (
    <div>
      <input type="text" value={value} onChange={handleInputChange} />
      
      {/* Render optimized components */}
      <DataProvider data={value}>
        {data => (
          <>
            {optimizedComponentA}
            {optimizedComponentB}
          </>
        )}
      </DataProvider>
    </div>
  );
};

export default App;
