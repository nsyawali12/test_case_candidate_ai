func checkPalindrome(strs text) bool {
  lastindx:= len(text) - 1
  for i:= 0; i < lastIndx/2 && (lastindx - i); i++ {
    if str[i] != str[lastidnx-1] {
      return false
    }
  }
  return true
}