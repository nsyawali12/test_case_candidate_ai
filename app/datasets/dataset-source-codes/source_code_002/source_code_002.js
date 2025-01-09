// Provided example is incorrect it should return [5] not [1,5]

function findOdd(arr) {
  const map = arr.reduce((_map, next) => {
    _map[next] = _map[next] || 0;
    _map[next] ++;
    return _map;
  }, {});

  const foundNumbers = [];
  Object.keys(map).forEach((key) => {
    const number = parseInt(key);
    const isAppearOddTimes = map[key] % 2 !== 0;
    if (isAppearOddTimes) {
      foundNumbers.push(number);
      if (foundNumbers.length === 2) {
        return foundNumbers;
      }
    }
  })
  return foundNumbers;
  
}