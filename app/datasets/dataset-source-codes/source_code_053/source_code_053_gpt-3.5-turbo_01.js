import React, { useState } from 'react';

const DynamicTable = () => {
  const [tableData, setTableData] = useState([]);

  const addRow = () => {
    const newRow = Array(tableData[0] ? tableData[0].length : 1).fill('');
    setTableData([...tableData, newRow]);
  };

  const addColumn = () => {
    const updatedTable = tableData.map(row => [...row, '']);
    setTableData(updatedTable);
  };

  const removeRow = (index) => {
    const updatedTable = tableData.filter((row, i) => i !== index);
    setTableData(updatedTable);
  };

  const removeColumn = (index) => {
    const updatedTable = tableData.map(row => row.filter((cell, i) => i !== index));
    setTableData(updatedTable);
  };

  const updateCell = (rowIndex, colIndex, value) => {
    const updatedTable = tableData.map((row, i) =>
      i === rowIndex ? row.map((cell, j) => (j === colIndex ? value : cell)) : row
    );
    setTableData(updatedTable);
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
                <td key={colIndex}>
                  <input
                    type="text"
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
          <tr>
            {tableData.length > 0 &&
              tableData[0].map((_, colIndex) => (
                <td key={colIndex}>
                  <button onClick={() => removeColumn(colIndex)}>Remove Column</button>
                </td>
              ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default DynamicTable;
