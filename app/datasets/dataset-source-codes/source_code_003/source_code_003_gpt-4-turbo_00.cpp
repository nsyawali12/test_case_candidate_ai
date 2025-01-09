#include <vector>

bool isValidSudoku(const std::vector<std::vector<int>>& board) {
    // Using arrays for row, column, and sub-grid checks
    int row[9][9] = {0}, col[9][9] = {0}, sub[3][3][9] = {0};

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            int num = board[i][j] - 1; // Adjusting number to be zero-indexed
            int k = (i / 3) * 3 + (j / 3); // Finding subgrid index

            // Check if this num has already been used in the row, column, or subgrid
            if (row[i][num]++ || col[j][num]++ || sub[i/3][j/3][num]++)
                return false;
        }
    }

    return true;
}
