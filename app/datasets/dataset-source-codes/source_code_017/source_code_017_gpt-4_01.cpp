#include <string>
using namespace std;

string middleCharacter(string s) {
    int len = s.length();
    int mid = len / 2;

    // If the length of the string is odd return the middle character
    if (len % 2 != 0)
        return string(1, s[mid]);
    // If the length of the string is even return the middle two characters
    else
        return s.substr(mid - 1, 2);
}
