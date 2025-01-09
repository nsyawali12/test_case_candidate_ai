function sortPlayers(players) {
    // Create a copy of the array to avoid mutating the original array
    const playersCopy = [...players];

    // Sort the array of player objects
    return playersCopy.sort((a, b) => {
        // Sort by score in descending order
        if (a.score > b.score) {
            return -1;
        } else if (a.score < b.score) {
            return 1;
        } else {
            // If scores are equal, sort by name in ascending order
            return a.name.localeCompare(b.name);
        }
    });
}
