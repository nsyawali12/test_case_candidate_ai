#include <iostream>
#include <string>
#include <algorithm>

bool isPalindrome(std::string str) {
    std::string original = str;
    std::reverse(str.begin(), str.end());
    return original == str;
}

int main() {
    std::string input;
    std::cout << "Enter a string: ";
    std::cin >> input;

    if (isPalindrome(input)) {
        std::cout << "The string is a palindrome." << std::endl;
    } else {
        std::cout << "The string is not a palindrome." << std::endl;
    }

    return 0;
}
