#include <string>

std::string getMiddle(const std::string& str) 
{
    int len = str.length();
    return (len % 2 == 0) ? str.substr(len / 2 - 1, 2) : str.substr(len / 2, 1);
}
