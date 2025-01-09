#include <vector>
#include <unordered_set>

bool isValidSudoku(std::vector<std::vector<int>>& board) {
    for (int i = 0; i < 9; ++i) {
        std::unordered_set<int> rowSet;
        std::unordered_set<int> colSet;
        std::unordered_set<int> subgridSet;
        
        for (int j = 0; j < 9; ++j) {
            // Check rows
            if (board[i][j] != 0 && !rowSet.insert(board[i][j]).second) {
                return false;
            }
            
            // Check columns
            if (board[j][i] != 0 && !colSet.insert(board[j][i]).second) {
                return false;
            }
            
            // Check subgrids
            int rowIdx = 3 * (i / 3) + j / 3;
            int colIdx = 3 * (i % 3) + j % 3;
            if (board[rowIdx][colIdx] != 0 && !subgridSet.insert(board[rowIdx][colIdx]).second) {
                return false;
            }
        }
    }
    
    return true;
}

// Example usage:
int main() {
    std::vector<std::vector<int>> board = {
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

    bool isValid = isValidSudoku(board);
    return 0;
}
