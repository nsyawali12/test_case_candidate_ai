// Assuming you have a data context set up to manage the data

import React, { useContext, useState, useEffect } from 'react';

const OptimizedComponent = () => {
  const { data, setData } = useContext(DataContext);
  const [optimizedData, setOptimizedData] = useState(data);

  // Update the optimizedData only when the original data changes
  useEffect(() => {
    setOptimizedData(data);
  }, [data]);

  const handleDataUpdate = (updatedData) => {
    // Update the original data and let the optimizedData re-render accordingly
    setData(updatedData);
  };

  return (
    <div>
      {/* Use the optimizedData instead of the original data */}
      {optimizedData.map(item => (
        <div key={item.id}>{item.name}</div>
      )}

      <button onClick={() => handleDataUpdate([...data, { id: data.length + 1, name: 'New Item' }])}>
        Add New Item
      </button>
    </div>
  );
};

export default OptimizedComponent;
