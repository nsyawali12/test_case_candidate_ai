#include <iostream>
#include <string>

std::string middleCharacter(const std::string& str) {
    int length = str.length();
    if (length % 2 == 0) {
        return str.substr((length / 2) - 1, 2);
    } else {
        return str.substr(length / 2, 1);
    }
}

int main() {
    std::string input1 = "abcde";
    std::string input2 = "abcdef";

    std::cout << "Middle character of '" << input1 << "': " << middleCharacter(input1) << std::endl;
    std::cout << "Middle character of '" << input2 << "': " << middleCharacter(input2) << std::endl;

    return 0;
}
