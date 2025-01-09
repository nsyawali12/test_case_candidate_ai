import React, { useState } from 'react';

function TableComponent() {
  const [columns, setColumns] = useState(['Column 1', 'Column 2', 'Column 3']);
  const [rows, setRows] = useState([
    ['Row 1, Cell 1', 'Row 1, Cell 2', 'Row 1, Cell 3']
  ]);

  const addRow = () => {
    const newRow = ['New Cell 1', 'New Cell 2', 'New Cell 3']; // Assuming 3 cells per row
    setRows([...rows, newRow]);
  };

  const removeSecondColumn = () => {
    setColumns(columns.filter((_, index) => index !== 1)); // Remove 2nd column
    setRows(rows.map(row => row.filter((_, index) => index !== 1)));
  };

  return (
    <div>
      <table>
        <thead>
          <tr>
            {columns.map((column, index) => <th key={index}>{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={addRow}>Add Row</button>
      <button onClick={removeSecondColumn}>Remove 2nd Column</button>
    </div>
  );
}

export default TableComponent;

for table update
function TableComponent({ data }) {
  // Assuming data is an array of objects with { id, name, value }
  return (
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item) => (
          <tr key={item.id}> {/* Unique key */}
            <td>{item.name}</td>
            <td>{item.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}