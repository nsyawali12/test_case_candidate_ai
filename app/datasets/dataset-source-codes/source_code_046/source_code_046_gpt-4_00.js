/**
 * Function to sort array of objects by score in descending order, then by name alphabetically (in case of a tie)
 * 
 * @param {Array} players - Array of player objects, each having a name and a score
 * @returns {Array} Returns a new sorted array of players
 */
function sortPlayers(players) {
    // Handle edge case when input array is empty
    if(players.length === 0) return [];

    // Create a new array for sorting to avoid mutating original array
    let sortedPlayers = [...players];

    // Use JavaScript's native sort() function to sort array
    sortedPlayers.sort((a, b) => {
        // Sort by score in descending order
        if(a.score > b.score) return -1;
        if(a.score < b.score) return 1;

        // In case of a tie, sort by name in alphabetical order
        if(a.name < b.name) return -1;
        if(a.name > b.name) return 1;

        // If both name and score are the same, keep original order
        return 0;
    });

    // Return the sorted array
    return sortedPlayers;
}
