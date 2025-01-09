function sortPlayersByScoreAndName(players) {
    if (players.length === 0) {
        return [];
    }

    // Sort the players array by score in descending order, then by name alphabetically
    const sortedPlayers = players.slice().sort((a, b) => {
        if (b.score - a.score !== 0) {
            return b.score - a.score; // Sort by score descending
        } else {
            return a.name.localeCompare(b.name); // If scores are equal, sort by name alphabetically
        }
    });

    return sortedPlayers;
}
