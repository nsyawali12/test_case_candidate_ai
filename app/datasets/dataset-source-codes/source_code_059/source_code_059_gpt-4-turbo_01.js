function compressObject(obj) {
    const result = {};

    function recurse(currObj, prefix = '') {
        for (const key in currObj) {
            // Check if the property belongs to the object and not its prototype.
            if (currObj.hasOwnProperty(key)) {
                const value = currObj[key];
                // Construct the new key.
                const newKey = prefix.length > 0 ? `${prefix}.${key}` : key;

                if (value && typeof value === 'object') {
                    // Recurse further if the value is a non-null object.
                    recurse(value, newKey);
                } else if (value !== null) {
                    // Only add non-null values to the result.
                    result[newKey] = value;
                }
            }
        }
    }

    recurse(obj);
    return result;
}
