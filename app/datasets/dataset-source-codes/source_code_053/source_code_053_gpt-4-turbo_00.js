import React, { useState } from 'react';

function DynamicTable() {
  const [tableData, setTableData] = useState([['']]); // Initial state with one cell

  // Add a new row to the table
  const addRow = () => {
    const newRow = new Array(tableData[0].length).fill(''); // Create a new row with empty cells
    setTableData([...tableData, newRow]);
  };

  // Add a new column to the table
  const addColumn = () => {
    const newTableData = tableData.map(row => [...row, '']); // Add a new cell to each row
    setTableData(newTableData);
  };

  // Remove a row from the table
  const removeRow = rowIndex => {
    if (tableData.length > 1) {
      const newTableData = tableData.filter((_, index) => index !== rowIndex);
      setTableData(newTableData);
    }
  };

  // Remove a column from the table
  const removeColumn = colIndex => {
    if (tableData[0].length > 1) {
      const newTableData = tableData.map(row => row.filter((_, index) => index !== colIndex));
      setTableData(newTableData);
    }
  };

  // Update cell data
  const updateCell = (rowIndex, colIndex, value) => {
    const updatedTableData = tableData.map((row, rIndex) => {
      if (rIndex === rowIndex) {
        return row.map((cell, cIndex) => (cIndex === colIndex ? value : cell));
      }
      return row;
    });
    setTableData(updatedTableData);
  };

  return (
    <div>
      <button onClick={addRow}>Add Row</button>
      <button onClick={addColumn}>Add Column</button>
      <table>
        <tbody>
          {tableData.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, colIndex) => (
                <td key={`${rowIndex}-${colIndex}`}>
                  <input 
                    value={cell}
                    onChange={(e) => updateCell(rowIndex, colIndex, e.target.value)} 
                  />
                </td>
              ))}
              <td>
                <button onClick={() => removeRow(rowIndex)}>Remove Row</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {tableData[0].length > 1 && <button onClick={() => removeColumn(tableData[0].length - 1)}>Remove Last Column</button>}
    </div>
  );
}

export default DynamicTable;
