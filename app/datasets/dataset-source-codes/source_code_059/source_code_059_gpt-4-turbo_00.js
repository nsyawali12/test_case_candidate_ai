function compressObject(obj) {
  const result = {};

  function helper(subObj, prefix) {
    Object.keys(subObj).forEach(key => {
      const value = subObj[key];
      const newKey = prefix ? `${prefix}.${key}` : key;
      if (value !== null && typeof value === 'object') {
        helper(value, newKey);
      } else if (value !== null) {
        result[newKey] = value;
      }
    });
  }

  helper(obj, '');
  return result;
}
