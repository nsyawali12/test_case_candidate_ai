#include <iostream>
#include <vector>

bool isValidSudoku(const std::vector<std::vector<int>>& board) {
    // Use 3 arrays to track row, column and sub-box presence of numbers
    int usedRows[9][9] = {0}, usedCols[9][9] = {0}, usedBoxes[9][9] = {0};

    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            int num = board[row][col] - 1; // adjust 1-9 index to 0-8
            int boxIndex = (row / 3) * 3 + col / 3; // compute box index

            // Check if number already used in row, column or box
            if (usedRows[row][num] || usedCols[col][num] || usedBoxes[boxIndex][num])
                return false;

            // Mark number as used in row, column and box
            usedRows[row][num] = 1;
            usedCols[col][num] = 1;
            usedBoxes[boxIndex][num] = 1;
        }
    }
    return true;
}

int main() {
    // Example of valid Sudoku
    std::vector<std::vector<int>> validSudoku = {
        {5,3,4,6,7,8,9,1,2},
        {6,7,2,1,9,5,3,4,8},
        {1,9,8,3,4,2,5,6,7},
        {8,5,9,7,6,1,4,2,3},
        {4,2,6,8,5,3,7,9,1},
        {7,1,3,9,2,4,8,5,6},
        {9,6,1,5,3,7,2,8,4},
        {2,8,7,4,1,9,6,3,5},
        {3,4,5,2,8,6,1,7,9}
    };

    // Example of invalid Sudoku (last two numbers in the last row are both 1)
    std::vector<std::vector<int>> invalidSudoku = {
        {5,3,4,6,7,8,9,1,2},
        {6,7,2,1,9,5,3,4,8},
        {1,9,8,3,4,2,5,6,7},
        {8,5,9,7,6,1,4,2,3},
        {4,2,6,8,5,3,7,9,1},
        {7,1,3,9,2,4,8,5,6},
        {9,6,1,5,3,7,2,8,4},
        {2,8,7,4,1,9,6,3,5},
        {3,4,5,2,8,6,1,7,1}
    };

    std::cout << "Valid Sudoku: " << isValidSudoku(validSudoku) << std::endl;  // Output: 1 (true)
    std::cout << "Invalid Sudoku: " << isValidSudoku(invalidSudoku) << std::endl; // Output: 0 (false)

    return 0;
}
