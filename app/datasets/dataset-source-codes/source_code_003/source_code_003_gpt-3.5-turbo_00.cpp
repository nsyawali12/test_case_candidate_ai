#include <iostream>
#include <vector>
using namespace std;

bool isValidSudoku(vector<vector<int>>& board) {
    vector<vector<bool>> row(9, vector<bool>(9, false));
    vector<vector<bool>> col(9, vector<bool>(9, false));
    vector<vector<bool>> subgrid(9, vector<bool>(9, false));

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == 0) continue;

            int num = board[i][j] - 1;
            int k = (i / 3) * 3 + j / 3;

            if (row[i][num] || col[j][num] || subgrid[k][num]) {
                return false;
            }

            row[i][num] = true;
            col[j][num] = true;
            subgrid[k][num] = true;
        }
    }

    return true;
}

int main() {
    vector<vector<int>> sudoku = {
        {5, 3, 4, 6, 7, 8, 9, 1, 2},
        {6, 7, 2, 1, 9, 5, 3, 4, 8},
        {1, 9, 8, 3, 4, 2, 5, 6, 7},
        {8, 5, 9, 7, 6, 1, 4, 2, 3},
        {4, 2, 6, 8, 5, 3, 7, 9, 1},
        {7, 1, 3, 9, 2, 4, 8, 5, 6},
        {9, 6, 1, 5, 3, 7, 2, 8, 4},
        {2, 8, 7, 4, 1, 9, 6, 3, 5},
        {3, 4, 5, 2, 8, 6, 1, 7, 9}
    };

    cout << isValidSudoku(sudoku) << endl;

    return 0;
}
