#include <vector>
#include <atlstr.h>

// Function to compute the temporary array
std::vector<int> computeTemporaryArray(ATL::CString pattern) {
    int size = pattern.GetLength();
    std::vector<int> lps(size, 0);

    int index = 0;
    for (int i = 1; i < size;) {
        if (pattern[i] == pattern[index]) {
            lps[i] = index + 1;
            index++;
            i++;
        } else {
            if (index != 0) {
                index = lps[index - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

// Main KMP function
std::vector<int> KMP(ATL::CString text, ATL::CString pattern) {
    std::vector<int> result;
    std::vector<int> lps = computeTemporaryArray(pattern);
    int i = 0;
    int j = 0;

    while (i < text.GetLength() && j < pattern.GetLength()) {
        if (text[i] == pattern[j]) {
            i++;
            j++;
        } else {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }

        // check if any match is found
        if (j == pattern.GetLength()) {
            result.push_back(i - j);
            j = lps[j - 1];
        }
    }

    return result;
}
