#include <atlstr.h>
#include <vector>

class KMP {
public:
    std::vector<int> findPatternIndices(const CString& text, const CString& pattern) {
        std::vector<int> indices;
        if (pattern.IsEmpty() || text.IsEmpty()) {
            return indices;
        }

        std::vector<int> lps = computeLPSArray(pattern);

        int i = 0, j = 0;
        while (i < text.GetLength()) {
            if (pattern[j] == text[i]) {
                j++;
                i++;
            }

            if (j == pattern.GetLength()) {
                indices.push_back(i - j);
                j = lps[j - 1];
            } else if (i < text.GetLength() && pattern[j] != text[i]) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }

        return indices;
    }

private:
    std::vector<int> computeLPSArray(const CString& pattern) {
        int len = 0;
        std::vector<int> lps(pattern.GetLength(), 0);
        
        int i = 1;
        while (i < pattern.GetLength()) {
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
};

void testKMP() {
    KMP kmp;
    
    CString text1 = "ABABDABACDABABCABAB";
    CString pattern1 = "ABABCABAB";
    std::vector<int> result1 = kmp.findPatternIndices(text1, pattern1);
    OutputDebugString("Input: text='ABABDABACDABABCABAB', pattern='ABABCABAB' -> Output: [");
    for (int i = 0; i < result1.size(); i++) {
        OutputDebugString(result1[i]);
        if (i < result1.size() - 1) {
            OutputDebugString(",");
        }
    }
    OutputDebugString("]\n");

    CString text2 = "AAAAA";
    CString pattern2 = "AAA";
    std::vector<int> result2 = kmp.findPatternIndices(text2, pattern2);
    OutputDebugString("Input: text='AAAAA', pattern='AAA' -> Output: [");
    for (int i = 0; i < result2.size(); i++) {
        OutputDebugString(result2[i]);
        if (i < result2.size() - 1) {
            OutputDebugString(",");
        }
    }
    OutputDebugString("]\n");
}
