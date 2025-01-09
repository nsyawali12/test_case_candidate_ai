#include <algorithm>
#include <string>

bool isPalindrome(const std::string& str) {
  // create a copy of original string and reverse it
  std::string temp = str;
  std::reverse(temp.begin(), temp.end());
  
  // if the reversed string equals the original, it is a palindrome
  return temp == str;
}

int main() {
  std::cout << (isPalindrome("radar") ? "true" : "false") << '\n';  // prints: true
  std::cout << (isPalindrome("hello") ? "true" : "false") << '\n';  // prints: false
  
  return 0;
}
