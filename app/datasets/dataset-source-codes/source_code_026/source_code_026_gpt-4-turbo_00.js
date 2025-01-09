import React, { useMemo, useState, useCallback } from 'react';

// Example component: Item
const Item = React.memo(({ itemData }) => {
  console.log('Rendering Item:', itemData.id); // Check which Item is being re-rendered
  return <div>{itemData.name}</div>;
});

// Parent component: ItemList
const ItemList = () => {
  const [items, setItems] = useState([
    { id: 1, name: 'Item 1' },
    { id: 2, name: 'Item 2' }
  ]);

  const [filter, setFilter] = useState('');

  // Handler to add a new item
  const addItem = useCallback(() => {
    setItems(prevItems => [...prevItems, { id: prevItems.length + 1, name: `Item ${prevItems.length + 1}` }]);
  }, []);

  // Handler to update filter
  const updateFilter = useCallback((e) => {
    setFilter(e.target.value);
  }, []);

  // Apply filter to items
  const filteredItems = useMemo(() => {
    console.log('Filtering items'); // Check when filtering happens
    return items.filter(item => item.name.toLowerCase().includes(filter.toLowerCase()));
  }, [items, filter]);

  return (
    <div>
      <input type="text" value={filter} onChange={updateFilter} placeholder="Filter items" />
      <button onClick={addItem}>Add Item</button>
      {filteredItems.map(item => (
        <Item key={item.id} itemData={item} />
      ))}
    </div>
  );
};

export default ItemList;
