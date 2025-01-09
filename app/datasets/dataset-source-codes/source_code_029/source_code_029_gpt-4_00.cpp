#include <atlstr.h>
#include <vector>

// Use ATL CString for string processing
typedef ATL::CString String;

// Function to compute the temporary array of the KMP algorithm
std::vector<int> computeTemporaryArray(String pattern) {
    int m = pattern.GetLength();
    std::vector<int> lpsArray(m, 0);  
    int index = 0;  
    for(int i = 1; i < m; ) {    
        if(pattern[i] == pattern[index]) {   
            lpsArray[i] = index + 1;  
            index++;  
            i++;  
        } else {
            if(index != 0){
                index = lpsArray[index-1];  
            } else {  
                lpsArray[i] = 0;  
                i++;  
            }  
        }  
    }  
    return lpsArray;  
}

// Function implementing KMP algorithm
std::vector<int> KMP(String text, String pattern) {
    std::vector<int> result;
    std::vector<int> lpsArray = computeTemporaryArray(pattern);  
    int i = 0;  
    int j = 0;  
    while(i < text.GetLength()) {  
        if(text[i] == pattern[j]) {  
            i++;  
            j++;  
        }
        if(j == pattern.GetLength()) {  
            result.push_back(i - j);  
            j = lpsArray[j-1];  
        }    
        else if(i < text.GetLength() && text[i] != pattern[j]) {  
            if(j != 0) {
                j = lpsArray[j-1];  
            }
            else {
                i++;  
            }  
        }
    }
    return result;  
}
