const checkIsPalindrome =(input: string)=>{
  const parsedInput = input.toLowerCase().replace(/[^a-zA-Z0-9\.?!]/g,'');
  const reversedString = parsedInput.split('').reverse().join('');
  return parsedInput === reversedString
}