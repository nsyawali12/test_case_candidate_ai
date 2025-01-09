import React, { useState } from 'react';

const DynamicTable = () => {
  const [tableData, setTableData] = useState([
    ['', ''],
    ['', '']
  ]);

  const handleAddRow = () => {
    const updatedTableData = [...tableData, Array(tableData[0].length).fill('')];
    setTableData(updatedTableData);
  };

  const handleAddColumn = () => {
    const updatedTableData = tableData.map(row => [...row, '']);
    setTableData(updatedTableData);
  };

  const handleRemoveRow = (index) => {
    const updatedTableData = tableData.filter((row, i) => i !== index);
    setTableData(updatedTableData);
  };

  const handleRemoveColumn = (index) => {
    const updatedTableData = tableData.map(row => row.filter((cell, i) => i !== index));
    setTableData(updatedTableData);
  };

  const handleCellChange = (rowIndex, colIndex, value) => {
    const updatedTableData = tableData.map((row, i) =>
      i === rowIndex ? row.map((cell, j) => (j === colIndex ? value : cell)) : row
    );
    setTableData(updatedTableData);
  };

  return (
    <div>
      <table>
        <tbody>
          {tableData.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, colIndex) => (
                <td key={colIndex}>
                  <input
                    type="text"
                    value={cell}
                    onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                  />
                </td>
              ))}
              <td>
                <button onClick={() => handleRemoveRow(rowIndex)}>Remove Row</button>
              </td>
            </tr>
          ))}
          <tr>
            <td>
              <button onClick={handleAddRow}>Add Row</button>
            </td>
            <td>
              <button onClick={handleAddColumn}>Add Column</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default DynamicTable;
