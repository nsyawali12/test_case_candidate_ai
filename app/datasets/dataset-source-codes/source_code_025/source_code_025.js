function isAnagram(s1, s2) {
  const a1 = s1.split('');
  const a2 = s2.split('');
  for (let c of a1) {
    const idx = a2.indexOf(c)
    if (idx == -1) {
      return false;
    }
    a2.splice(idx, 1)
  }
  return a2.length == 0
}