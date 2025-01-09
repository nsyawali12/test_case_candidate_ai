function matchUsers(users) {
    let pairs = [];
    let used = new Set();

    // Iterate through users to try and match them
    for (let i = 0; i < users.length; i++) {
        if (used.has(i)) continue;

        let maxScore = -1;
        let bestPair = null;

        for (let j = i + 1; j < users.length; j++) {
            if (used.has(j)) continue;

            let score = calculateSimilarityScore(users[i], users[j]);
            if (score > maxScore) {
                maxScore = score;
                bestPair = j;
            }
        }

        if (bestPair != null) {
            pairs.push([users[i].name, users[bestPair].name]);
            used.add(i);
            used.add(bestPair);
        }
    }

    return pairs;
}

function calculateSimilarityScore(user1, user2) {
    let commonSkills = 0;
    let totalRatingDifference = 0;

    for (let skill in user1.skills) {
        if (user2.skills.hasOwnProperty(skill)) {
            commonSkills++;
            totalRatingDifference += Math.abs(user1.skills[skill] - user2.skills[skill]);
        }
    }

    if (commonSkills === 0) {
        return -1; // No common skills
    }

    // Score is calculated based on number of common skills and how close their ratings are.
    return commonSkills * 100 - totalRatingDifference;
}

// Example usage:
const users = [
    { name: 'Alice', skills: { Java: 5, C++: 3 } },
    { name: 'Bob', skills: { Java: 6, Python: 4 } },
    { name: 'Charlie', skills: { Python: 5, C++: 5 } }
];

console.log(matchUsers(users));
