function findBestPairs(users) {
    const pairings = [];
  
    const sortedUsers = users.sort((a, b) => {
        const avgRatingA = calculateAvgSkillRating(a);
        const avgRatingB = calculateAvgSkillRating(b);
        return avgRatingB - avgRatingA;
    });
  
    while (sortedUsers.length > 1) {
        const user = sortedUsers.shift();
        const bestPair = findBestPair(user, sortedUsers);
        if (bestPair) {
            pairings.push([user.name, bestPair.name]);
            sortedUsers.splice(sortedUsers.indexOf(bestPair), 1);
        }
    }
  
    return pairings;
}

function findBestPair(user, users) {
    let bestPair = null;
    let maxSimilarity = 0;
  
    for (let otherUser of users) {
        const similarity = calculateSkillSimilarity(user, otherUser);
        
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestPair = otherUser;
        }
    }
  
    return bestPair;
}

function calculateSkillSimilarity(user1, user2) {
    let commonSkills = 0;
    let totalSimilarity = 0;
  
    for (let skill in user1.skills) {
        if (user2.skills.hasOwnProperty(skill)) {
            commonSkills++;
            totalSimilarity += Math.abs(user1.skills[skill] - user2.skills[skill]);
        }
    }
  
    return commonSkills === 0 ? 0 : totalSimilarity / commonSkills;
}

function calculateAvgSkillRating(user) {
    const totalRating = Object.values(user.skills).reduce((acc, val) => acc + val, 0);
    const numSkills = Object.keys(user.skills).length;
  
    return numSkills === 0 ? 0 : totalRating / numSkills;
}

const users = [
    { name: 'Alice', skills: { Java: 5, Cpp: 3 } },
    { name: 'Bob', skills: { Java: 6, Python: 4 } },
    { name: 'Charlie', skills: { Python: 5, Cpp: 5 } }
];

const pairs = findBestPairs(users);
console.log(pairs);
