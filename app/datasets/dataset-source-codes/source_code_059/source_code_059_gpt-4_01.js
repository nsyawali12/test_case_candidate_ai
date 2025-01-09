function flatten(obj, prefix = '', res = {}) {
  // iterate through each property of the object
  for (let key in obj) {
    if (obj.hasOwnProperty(key)) {
      // if property value is an object and not null, recursively flatten it
      if (typeof obj[key] === 'object' && obj[key] !== null) {
        flatten(obj[key], prefix + key + '.', res);
      } else if (obj[key] !== null) {
        // if property value is not null, add it to result
        res[prefix + key] = obj[key];
      }
    }
  }
  return res;
}

let input = { a: 1, b: { c: null, d: { e: 2 } }, f: { g: null } }
console.log(flatten(input));
