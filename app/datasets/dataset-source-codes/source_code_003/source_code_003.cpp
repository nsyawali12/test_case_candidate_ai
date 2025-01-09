#include <iostream>


using namespace std;

const int rows = 9;
const int columns = 9;

bool isPassedGridIsSudoku(int (*grid)[columns])
{
    int desired_sum = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9;
    int received_sum;
    int isSudoku = true;

    // First condition
    // Loop through all rows

    for (int i = 0; i < rows; i++)
    {
        received_sum = 0;
        // Get specific row and loop through all numbers inside it
        for (int j = 0; j < columns; j++)
        {
           received_sum += grid[i][j];
        }
        if (received_sum != desired_sum)
        {
            isSudoku = false;
            break;
        }
    }
    // Second condition
    // Loop through all columns
    for (int i = 0; i < columns; i++)
    {
        received_sum = 0;
        // Get specific columns and loop through all numbers inside it
        for (int j = 0; j < rows; j++)
        {
           received_sum += grid[j][i];
        }
        if (received_sum != desired_sum)
        {
            isSudoku = false;
            break;
        }
    }

    // Third condition each minor sudoku need to be between 1-9
    for (int i = 0; i < rows; i+=3)
    {
        for (int k = i; k < i+3; k++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (i % 3 == 0 && j % 3 == 0)
                {
                    if (received_sum != desired_sum)
                    {
                        isSudoku = false;
                        break;
                    }
                    // reset
                    received_sum = 0;
                }
                    
                received_sum += grid[k][j];
            }

            if (received_sum != desired_sum)
            {
                isSudoku = false;
                break;
            }
        }
            
    }



    return isSudoku;
}


int main() {

    
    int grid[rows][columns] = {
        {1,2,3,4,5,6,7,8,9},
        {1,2,3,4,5,6,7,8,9},
        {1,2,3,4,5,6,7,8,9},
   