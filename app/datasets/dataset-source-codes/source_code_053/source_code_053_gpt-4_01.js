import React, { Component } from 'react';

class DynamicTable extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [[""]],
    };
  }
  
  // Adds a new row
  handleAddRow = () => {
    this.setState({
      data: [...this.state.data, [""]]
    });
  };
  
  // Removes a row
  handleRemoveRow = (rowIndex) => {
    this.setState({
      data: this.state.data.filter((row, index) => index !== rowIndex)
    });
  };

  // Adds a new column
  handleAddColumn = () => {
    this.setState({
      data: this.state.data.map(row => [...row, ""])
    });
  };
  
  // Removes a column
  handleRemoveColumn = (colIndex) => {
    this.setState({
      data: this.state.data.map(row => row.filter((cell, index) => index !== colIndex))
    });
  };

  // Updates the data of a single cell
  updateData = (rowIndex, columnIndex, event) => {
    let copyData = [...this.state.data];
    copyData[rowIndex][columnIndex] = event.target.value;
    this.setState({data: copyData});
  };

  render() {
    return (
      <div>
        <button onClick={this.handleAddRow}>Add Row</button>
        <button onClick={this.handleAddColumn}>Add Column</button>
        <table>
          <tbody>
            {this.state.data.map((rowData, rowIndex) => (
              <tr key={rowIndex}>
                {rowData.map((cellData, colIndex) => (
                  <td key={colIndex}>
                    <input value={cellData} onChange={(event) => this.updateData(rowIndex,colIndex,event)}/>
                  </td>
                ))}
                <td>
                  <button onClick={() => { this.handleRemoveRow(rowIndex)}}> Remove Row </button>
                  {rowData.map((data, colIndex) => (
                    <button key={colIndex} onClick={() => { this.handleRemoveColumn(colIndex)}}>Remove Column</button>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
}

export default DynamicTable;
