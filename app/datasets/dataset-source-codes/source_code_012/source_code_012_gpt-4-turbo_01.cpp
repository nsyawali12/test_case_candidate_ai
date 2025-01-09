#include <iostream>
#include <string>

bool isPalindrome(const std::string& str) {
    int left = 0, right = str.length() - 1;
    
    // Compare characters moving towards the center
    while (left < right) {
        if (str[left] != str[right]) {
            return false; // Not a palindrome if mismatch occurs
        }
        left++;
        right--;
    }
    return true; // It is a palindrome if no mismatches occurred
}

int main() {
    std::string input;
    std::cout << "Enter a string: ";
    std::cin >> input;

    // Output the result
    std::cout << (isPalindrome(input) ? "true" : "false") << std::endl;

    return 0;
}
