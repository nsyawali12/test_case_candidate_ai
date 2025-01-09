import React, { useMemo, useState, useCallback } from 'react';

// A generic component that accepts an item and a handler to update it
const ListItem = React.memo(({ item, onUpdate }) => {
    console.log(`Rendering ListItem: ${item.id}`);
    return (
        <div>
            {item.name}
            <button onClick={() => onUpdate(item.id, `${item.name} Updated`)}>Update</button>
        </div>
    );
});

const List = ({ items, setItems }) => {
    // useCallback to prevent unnecessary re-renders of ListItem
    const handleUpdate = useCallback((id, newName) => {
        setItems(items =>
            items.map(item =>
                item.id === id ? { ...item, name: newName } : item
            )
        );
    }, [setItems]);

    return (
        <div>
            {items.map(item => (
                <ListItem key={item.id} item={item} onUpdate={handleUpdate} />
            ))}
        </div>
    );
};

// Main component, managing the state and providing handlers
const App = () => {
    const [items, setItems] = useState([
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' },
        { id: 3, name: 'Item 3' },
    ]);

    // useMemo to prevent unnecessary calculations/re-renders
    const memoizedItems = useMemo(() => items, [items]);

    return (
        <div>
            <List items={memoizedItems} setItems={setItems} />
        </div>
    );
};

export default App;
