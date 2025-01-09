function compressObject(obj, parentKey = '', result = {}) 
{
    for (let key in obj) 
    {
        if (obj.hasOwnProperty(key)) 
        {
            const nestedKey = parentKey ? `${parentKey}.${key}` : key;
            if (obj[key] == null) 
            {
                
                continue;
            }
            else if (typeof obj[key] === 'object' && !Array.isArray(obj[key])) 
            {
               
                compressObject(obj[key], nestedKey, result);
            }
            else 
            {               
                result[nestedKey] = obj[key];
            }
        }
    }
    return result;
}

const input = { a: 1, b: { c: null, d: { e: 2 } }, f: { g: null } };
const output = compressObject(input);
console.log(output);