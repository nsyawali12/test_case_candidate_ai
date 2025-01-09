// Function to sort an array of players by 'score' and 'name'
function sortPlayers(players) {
    // Check for edge case where the input array is empty
    if (!Array.isArray(players) || !players.length) {
        return [];
    }

    // Using the Array.prototype.sort() method on a duplicate array to avoid mutation of the original array.
    // First, create a duplicate array using the Array.prototype.slice() method
    let sortedPlayers = players.slice();

    // Now, sort the duplicate array
    sortedPlayers.sort((a, b) => {
        // First, sort by 'score' in descending order
        if (a.score > b.score) {
            return -1;
        } else if (a.score < b.score) {
            return 1;
        } else {
            // If scores are equal, sort by 'name' alphabetically (in ascending order)
            if (a.name < b.name) {
                return -1;
            } else if (a.name > b.name) {
                return 1;
            } else {
                return 0;
            }
        }
    });

    // Return the sorted array
    return sortedPlayers;
}
