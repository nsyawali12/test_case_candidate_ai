import React, { useState } from 'react';

function DynamicTable() {
  // Initialize state for rows, columns and table data
  const [rows, setRows] = useState(1);
  const [columns, setColumns] = useState(1);
  const [data, setData] = useState([[""]]);

  // Handler to add a row
  const addRow = () => {
    setRows(rows + 1);
    setData([...data, new Array(columns).fill("")]);
  };

  // Handler to remove a row
  const removeRow = (rowIndex) => {
    if (rows > 1) {
      const updatedData = data.filter((_, index) => index !== rowIndex);
      setRows(rows - 1);
      setData(updatedData);
    }
  };

  // Handler to add a column
  const addColumn = () => {
    setColumns(columns + 1);
    const updatedData = data.map(row => [...row, ""]);
    setData(updatedData);
  };

  // Handler to remove a column
  const removeColumn = (colIndex) => {
    if (columns > 1) {
      const updatedData = data.map(row => row.filter((_, index) => index !== colIndex));
      setColumns(columns - 1);
      setData(updatedData);
    }
  };

  // Handler to update data in cell
  const updateCell = (rowIndex, colIndex, value) => {
    const updatedData = data.map((row, rIndex) => (
      rIndex === rowIndex ? row.map((cell, cIndex) => (
        cIndex === colIndex ? value : cell
      )) : row
    ));
    setData(updatedData);
  };

  return (
    <div>
      <button onClick={addRow}>Add Row</button>
      <button onClick={addColumn}>Add Column</button>
      <table>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, colIndex) => (
                <td key={colIndex}>
                  <input
                    type="text"
                    value={cell}
                    onChange={e => updateCell(rowIndex, colIndex, e.target.value)}
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
      {columns > 0 && (
        <div>
          {[...Array(columns).keys()].map(colIndex => (
            <button key={colIndex} onClick={() => removeColumn(colIndex)}>
              Remove Column {colIndex + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default DynamicTable;
