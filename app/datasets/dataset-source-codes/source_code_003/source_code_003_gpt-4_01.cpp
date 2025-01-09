#include <vector>

bool isValidSudoku(std::vector<std::vector<int> > &grid) {
  // Declare some variables to use for checks
  int rows[9][9] = {0}, cols[9][9] = {0}, boxes[9][9] = {0};

  // Iterate over the entire grid
  for(int i = 0; i < 9; ++i) {
    for(int j = 0; j < 9; ++j) {
      int num = grid[i][j] - 1; // Current number (minus 1 to fit 0-8 index)
      int k = i / 3 * 3 + j / 3; // Current box
      
      // If number is already in current row, col or box
      if(rows[i][num] || cols[j][num] || boxes[k][num])
        return false;
      
      // Place number in current row, col and box
      rows[i][num] = cols[j][num] = boxes[k][num] = 1;
    }
  }
  
  // If no problems found, it's a valid sudoku
  return true;
}
