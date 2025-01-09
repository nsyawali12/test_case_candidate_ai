#include <atlstr.h> // include ATL library for string processing

class Solution {
public:
    CAtlArray<int> findPatternOccurrences(const CString& text, const CString& pattern) {
        CAtlArray<int> occurrences;
        if (pattern.IsEmpty()) {
            return occurrences;
        }

        CAtlArray<int> lps = calculateLPS(pattern); // calculate Longest Prefix Suffix array

        int i = 0, j = 0;
        while (i < text.GetLength()) {
            if (pattern[j] == text[i]) {
                i++;
                j++;
            }

            if (j == pattern.GetLength()) {
                occurrences.Add(i - j);
                j = lps[j - 1];
            } else if (i < text.GetLength() && pattern[j] != text[i]) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }

        return occurrences;
    }

private:
    CAtlArray<int> calculateLPS(const CString& pattern) {
        CAtlArray<int> lps(pattern.GetLength());
        int len = 0;
        lps.SetAt(0, 0);

        int i = 1;
        while (i < pattern.GetLength()) {
            if (pattern[i] == pattern[len]) {
                len++;
                lps.SetAt(i, len);
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps.SetAt(i, 0);
                    i++;
                }
            }
        }

        return lps;
    }
};

int main() {
    Solution solution;

    // Test case 1
    CString text1 = "ABABDABACDABABCABAB";
    CString pattern1 = "ABABCABAB";
    CAtlArray<int> result1 = solution.findPatternOccurrences(text1, pattern1);
    for (int i = 0; i < result1.GetCount(); i++) {
        printf("%d ", result1.GetAt(i));
    }
    printf("\n");

    // Test case 2
    CString text2 = "AAAAA";
    CString pattern2 = "AAA";
    CAtlArray<int> result2 = solution.findPatternOccurrences(text2, pattern2);
    for (int i = 0; i < result2.GetCount(); i++) {
        printf("%d ", result2.GetAt(i));
    }
    printf("\n");

    return 0;
}
