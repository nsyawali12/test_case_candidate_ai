#include <iostream>
#include <string>

bool isPalindrome(const std::string &str) {
    int left = 0; // Start pointer
    int right = str.length() - 1; // End pointer

    while (left < right) {
        if (str[left] != str[right]) {
            return false; // Not a palindrome if mismatch happens
        }
        left++;  // Move left pointer right
        right--; // Move right pointer left
    }
    return true; // It is a palindrome if no mismatches occur
}

int main() {
    std::string input = "radar";
    std::cout << (isPalindrome(input) ? "true" : "false") << std::endl;

    input = "hello";
    std::cout << (isPalindrome(input) ? "true" : "false") << std::endl;

    return 0;
}
