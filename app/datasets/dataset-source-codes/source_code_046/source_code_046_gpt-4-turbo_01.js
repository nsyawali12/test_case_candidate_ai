function sortPlayers(players) {
    // Create a copy of the array to avoid mutating the original array
    const playersCopy = [...players];

    // Use .sort() method to sort the array
    return playersCopy.sort((a, b) => {
        // Compare scores in descending order
        if (a.score !== b.score) {
            return b.score - a.score;
        }
        // If scores are equal, compare names in ascending order
        return a.name.localeCompare(b.name);
    });
}
