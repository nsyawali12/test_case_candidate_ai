import React, { Component } from 'react';

class DynamicTable extends Component {
  constructor(props) {
    super(props);
    this.state = {
      rows: 0,
      columns: 0,
      text: [],
    };

    this.addRow = this.addRow.bind(this);
    this.addColumn = this.addColumn.bind(this);
    this.removeRow = this.removeRow.bind(this);
    this.removeColumn = this.removeColumn.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  addRow() {
    // Increase the number of rows and update the state
    this.setState((prevState) => ({ rows: prevState.rows + 1 }));
  }

  addColumn() {
    // Increase the number of columns and update the state
    this.setState((prevState) => ({ columns: prevState.columns + 1 }));
  }

  removeRow() {
    // Decrease the number of rows and update the state
    this.setState((prevState) => ({ rows: prevState.rows - 1 }));
  }

  removeColumn() {
    // Decrease the number of columns and update the state
    this.setState((prevState) => ({ columns: prevState.columns - 1 }));
  }
  
  handleChange(event, row, column) {
    // Update the text state upon changes
    let newText = [...this.state.text];
    newText[row] = newText[row] || [];
    newText[row][column] = event.target.value;
    this.setState({ text: newText });
  }

  render() {
    return (
      <div>
        <button onClick={this.addRow}>Add Row</button>
        <button onClick={this.addColumn}>Add Column</button>
        <button onClick={this.removeRow}>Remove Row</button>
        <button onClick={this.removeColumn}>Remove Column</button>
        <table>
          {Array(this.state.rows).fill(null).map((row, rowIndex) => (
            <tr>
              {Array(this.state.columns).fill(null).map((column, columnIndex) => (
                <td>
                  <input 
                    value={this.state.text[rowIndex] && this.state.text[rowIndex][columnIndex] || ''}
                    onChange={(e) => this.handleChange(e, rowIndex, columnIndex)}
                  />
                </td>
              ))}
            </tr>
          ))}
        </table>
      </div>
    );
  }
}

export default DynamicTable;
