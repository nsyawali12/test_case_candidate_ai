#include <iostream>
#include <string>

std::string getMiddleCharacters(std::string str) {
    int length = str.length();
    if (length % 2 == 0) {
        return str.substr(length / 2 - 1, 2);
    } else {
        return str.substr(length / 2, 1);
    }
}

int main() {
    std::string input1 = "abcde";
    std::string input2 = "abcdef";

    // Test case 1
    std::cout << "Middle characters of '" << input1 << "': " << getMiddleCharacters(input1) << std::endl;

    // Test case 2
    std::cout << "Middle characters of '" << input2 << "': " << getMiddleCharacters(input2) << std::endl;

    return 0;
}
