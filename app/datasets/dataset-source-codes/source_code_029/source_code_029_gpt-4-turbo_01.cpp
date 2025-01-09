#include <atlstr.h> // Include ATL String classes
#include <vector>

// Function to create the longest prefix suffix (lps) array
void computeLPSArray(CString pattern, int M, std::vector<int> &lps) {
    int length = 0;  // Length of the previous longest prefix suffix
    int i = 1;
    lps[0] = 0; // lps[0] is always 0

    // Loop calculates lps[i] for i = 1 to M-1
    while (i < M) {
        if (pattern[i] == pattern[length]) {
            length++;
            lps[i] = length;
            i++;
        } else { // (pattern[i] != pattern[length])
            if (length != 0) {
                length = lps[length - 1];
            } else { // if (length == 0)
                lps[i] = 0;
                i++;
            }
        }
    }
}

// KMP algorithm to find all occurrences of the pattern in the text
std::vector<int> KMP_Search(CString text, CString pattern) {
    int M = pattern.GetLength();
    int N = text.GetLength();
    std::vector<int> lps(M), results;

    // Compute the lps array
    computeLPSArray(pattern, M, lps);

    int i = 0;  // Index for text
    int j = 0;  // Index for pattern
    while (i < N) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }

        if (j == M) {
            results.push_back(i - j);
            j = lps[j - 1];
        } else if (i < N && pattern[j] != text[i]) { 
            // Mismatch after j matches
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }

    return results;
}

#include <iostream>

// Main Function to demonstrate above function
int main() {
    CString text = _T("ABABDABACDABABCABAB");
    CString pattern = _T("ABABCABAB");
    auto result = KMP_Search(text, pattern);

    std::cout << "[";
    for(size_t i = 0; i < result.size(); i++) {
        std::cout << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    return 0;
}
