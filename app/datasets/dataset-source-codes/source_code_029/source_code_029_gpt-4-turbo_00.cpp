#include <atlstr.h>  // Include ATL string classes
#include <vector>

// Function to create the lps (Longest Prefix Suffix) array for KMP algorithm
std::vector<int> computeLPSArray(const CString& pattern) {
    int M = pattern.GetLength();
    std::vector<int> lps(M, 0);
    int len = 0;
    int i = 1;
    
    // Compute lps array
    while (i < M) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

// KMP search algorithm
std::vector<int> KMPSearch(const CString& text, const CString& pattern) {
    int M = pattern.GetLength();
    int N = text.GetLength();
    std::vector<int> lps = computeLPSArray(pattern);
    std::vector<int> results;

    int i = 0;  // index for text
    int j = 0;  // index for pattern
    while (i < N) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }

        if (j == M) {
            results.push_back(i - j);
            j = lps[j - 1];
        } else if (i < N && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }

    return results;
}

// Example usage
int main() {
    CString text = _T("ABABDABACDABABCABAB");
    CString pattern = _T("ABABCABAB");
    auto results = KMPSearch(text, pattern);
    for (int index : results) {
        std::cout << index << " ";
    }

    std::cout << std::endl;
    return 0;
}
