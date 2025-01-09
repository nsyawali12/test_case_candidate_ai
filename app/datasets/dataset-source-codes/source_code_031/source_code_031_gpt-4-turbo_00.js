function matchUsersBySkills(users) {
    const n = users.length;
    let bestPairAverage = -1;
    let bestPairs = [];

    // This function will check if two users have at least one skill in common and calculate the average skill rating
    function checkAndCalcAvgSkillRating(user1, user2) {
        let commonSkills = [];
        let totalRating = 0;
        // Identify common skills between two users
        for (let skill in user1.skills) {
            if (skill in user2.skills) {
                commonSkills.push(skill);
                totalRating += Math.abs(user1.skills[skill] - user2.skills[skill]);
            }
        }
        // Calculate the adjusted average skill rating only if there is common skill
        if (commonSkills.length > 0) {
            return totalRating / commonSkills.length;
        }
        return null;
    }

    // Function to find best pairs of users based on their skills
    function findPairs(used, currentPairs, currentIndex) {
        if (currentIndex >= n) {
            let currentAverage = currentPairs.reduce((acc, pair) => acc + pair.avg, 0) / (currentPairs.length || 1);
            if (currentPairs.length > 0 && (bestPairAverage === -1 || currentAverage < bestPairAverage)) {
                bestPairAverage = currentAverage;
                bestPairs = currentPairs.map(pair => [pair.user1.name, pair.user2.name]);
            }
            return;
        }

        if (used[currentIndex]) {
            findPairs(used, currentPairs, currentIndex + 1);
        } else {
            used[currentIndex] = true;
            // try pairing current user with any other user
            for (let i = currentIndex + 1; i < n; i++) {
                if (!used[i]) {
                    let avg = checkAndCalcAvgSkillRating(users[currentIndex], users[i]);
                    if (avg !== null) {
                        used[i] = true;
                        findPairs(used, [...currentPairs, { user1: users[currentIndex], user2: users[i], avg }], currentIndex + 1);
                        used[i] = false;
                    }
                }
            }
            used[currentIndex] = false;
            findPairs(used, currentPairs, currentIndex + 1);
        }
    }

    // Start recursive pairing
    findPairs(Array(n).fill(false), [], 0);

    return bestPairs;
}

// Example usage:
const users = [
    { name: 'Alice', skills: { Java: 5, C++: 3 } },
    { name: 'Bob', skills: { Java: 6, Python: 4 } },
    { name: 'Charlie', skills: { Python: 5, C++: 5 } }
];
console.log(matchUsersBySkills(users));
