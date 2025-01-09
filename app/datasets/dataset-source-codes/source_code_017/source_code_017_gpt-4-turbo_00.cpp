#include <iostream>
#include <string>
using namespace std;

string getMiddle(string s) {
    int n = s.length();
    if (n % 2 == 1) // If the length is odd
        return s.substr(n / 2, 1);
    else // If the length is even
        return s.substr(n / 2 - 1, 2);
}

int main() {
    // Test cases
    cout << getMiddle("abcde") << endl;   // Should return "c"
    cout << getMiddle("abcdef") << endl;  // Should return "cd"
    return 0;
}
