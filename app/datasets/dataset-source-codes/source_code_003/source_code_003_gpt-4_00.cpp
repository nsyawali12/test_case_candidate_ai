#include <vector>
#include <iostream>

bool isValidSudoku(std::vector<std::vector<int>>& board) {
    // Initialize rows, columns and boxes
    int rows[9][9] = {0}, columns[9][9] = {0}, boxes[3][3][9] = {0};

    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            
            int num = board[row][col] - 1; // subtract 1 to keep zero indexing
            if (rows[row][num] || columns[col][num] || boxes[row / 3][col / 3][num]){
                return false; // return false if a repetition is found
            }

            // else mark as filled
            rows[row][num] = columns[col][num] = boxes[row / 3][col / 3][num] = 1;
        }
    }
    return true; // return true if no repetition was found
}
