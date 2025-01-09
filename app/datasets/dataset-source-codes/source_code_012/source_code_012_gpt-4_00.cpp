#include <string>
#include <algorithm>

bool checkPalindrome(std::string str) {
    //Create a copy of the original string and reverse it
    std::string revStr = str;
    std::reverse(revStr.begin(), revStr.end());

    //Compare the original string with its reversed version
    if (str == revStr) {
        return true;
    } else {
        return false;
    }
}

int main() {
    cout << checkPalindrome("radar"); // returns 1 (true)
    cout << checkPalindrome("hello"); // returns 0 (false)
    return 0;
}
