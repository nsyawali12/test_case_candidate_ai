function matchSkills(users) {
    let pairs = [];
    while (users.length > 1) {
        let maxAvg = -1, candidate = -1;
        for (let i = 1; i < users.length; i++) {
            let commonRating = calcCommonRating(users[0], users[i]);
            if (commonRating > maxAvg) {
                maxAvg = commonRating;
                candidate = i;
            }
        }
        if (candidate !== -1) {
            pairs.push([users[0].name, users[candidate].name]); //Pair the users with similar common skill and rating
            users.splice(candidate, 1); // Remove the candidate from users array
        }
        users.shift(); // Remove the first user from users array
    }
    return pairs;
}

function calcCommonRating(user1, user2) {
    let commonSkillCount = 0, totalRating = 0;
    for (let skill in user1.skills) {
        if (user2.skills[skill]) {
            commonSkillCount++;
            totalRating += (user1.skills[skill] + user2.skills[skill]) / 2.0; // Calculate the average rating
        }
    }
    return commonSkillCount === 0 ? -1 : totalRating / commonSkillCount;
}
