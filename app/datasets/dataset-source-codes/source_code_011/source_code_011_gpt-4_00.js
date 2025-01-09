function removeDuplicates(str) {
    let result = '';

    // We use a Set (which only allows unique values) to remove duplicates
    let unique = new Set([...str]);

    // We iterate over the set and create a new string
    for(let char of unique) {
        result += char;
    }

    return result;
}
