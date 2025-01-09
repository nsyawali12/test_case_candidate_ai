function sortPlayers(arr) {
    // Check if the input array is empty
    if (arr.length === 0) {
        return [];
    }

    // Sort the array by score descending and by name alphabetically
    const sortedArray = arr.slice().sort((a, b) => {
        if (b.score !== a.score) {
            return b.score - a.score; // Sort by score descending
        } else {
            return a.name.localeCompare(b.name); // Then sort alphabetically by name
        }
    });

    return sortedArray;
}

// Example usage
const players = [
    { name: 'Alice', score: 10 },
    { name: 'Bob', score: 8 },
    { name: 'Charlie', score: 10 }
];
console.log(sortPlayers(players));
