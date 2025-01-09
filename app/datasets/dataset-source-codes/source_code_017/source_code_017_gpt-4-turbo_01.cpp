#include <iostream>
#include <string>

std::string getMiddle(const std::string& str) {
    int len = str.length();
    int mid = len / 2;
    
    if (len % 2 == 1) {
        // Return the middle character for odd length strings
        return std::string(1, str[mid]);
    } else {
        // Return the middle two characters for even length strings
        return str.substr(mid - 1, 2);
    }
}

int main() {
    std::string input1 = "abcde";
    std::string input2 = "abcdef";
    std::cout << "Middle of '" << input1 << "': " << getMiddle(input1) << '\n';
    std::cout << "Middle of '" << input2 << "': " << getMiddle(input2) << '\n';
    return 0;
}
