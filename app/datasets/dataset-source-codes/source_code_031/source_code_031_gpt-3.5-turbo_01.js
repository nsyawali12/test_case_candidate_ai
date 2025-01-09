function matchUsers(users) {
    let pairings = [];
    
    while (users.length > 1) {
        let bestPair = [];
        let bestDiff = 0;
        
        // Find the pair with the highest average similar skill rating
        for (let i = 0; i < users.length-1; i++) {
            for (let j = i+1; j < users.length; j++) {
                let commonSkills = Object.keys(users[i].skills).filter(skill => users[j].skills[skill] !== undefined);
                
                if (commonSkills.length > 0) {
                    let totalRating = 0;
                    commonSkills.forEach(skill => {
                        totalRating += Math.abs(users[i].skills[skill] - users[j].skills[skill]);
                    });
                    
                    let diff = totalRating / commonSkills.length;
                    
                    if (diff < bestDiff || bestPair.length === 0) {
                        bestPair = [users[i].name, users[j].name];
                        bestDiff = diff;
                    }
                }
            }
        }

        // Remove the paired users from the pool
        pairings.push(bestPair);
        users = users.filter(user => user.name !== bestPair[0] && user.name !== bestPair[1]);
    }

    return pairings;
}

const users = [
    { name: 'Alice', skills: { Java: 5, Cpp: 3 } },
    { name: 'Bob', skills: { Java: 6, Python: 4 } },
    { name: 'Charlie', skills: { Python: 5, Cpp: 5 } }
];

console.log(matchUsers(users));
