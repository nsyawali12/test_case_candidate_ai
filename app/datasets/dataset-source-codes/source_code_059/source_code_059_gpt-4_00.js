function flatten(object) {
    const result = {};

    // helper function to recursively flatten the object
    function recurse(obj, current) {
        for(let key in obj) {
            let value = obj[key];
            let newKey = (current ? current + '.' + key : key);  // formed 'dot' notation key

            if(value && typeof value === 'object') {
                recurse(value, newKey);
            } else if(value !== null) {
              result[newKey] = value;
            }
        }
    }

    recurse(object, '');
    return result;
}

let input = { a: 1, b: { c: null, d: { e: 2 } }, f: { g: null } };
console.log(flatten(input));  // outputs: {'a': 1, 'b.d.e': 2}
