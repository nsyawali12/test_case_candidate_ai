function compressObject(obj, parentKey = '', result = {}) {
    for (let key in obj) {
        let newKey = parentKey ? `${parentKey}.${key}` : key;

        if (obj[key] !== null && typeof obj[key] === 'object') {
            compressObject(obj[key], newKey, result);
        } else if (obj[key] !== null) {
            result[newKey] = obj[key];
        }
    }
    
    return result;
}

const input = { a: 1, b: { c: null, d: { e: 2 } }, f: { g: null } };
const output = compressObject(input);
console.log(output);
