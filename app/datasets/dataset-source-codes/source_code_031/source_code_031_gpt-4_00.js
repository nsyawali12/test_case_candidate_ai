function pairUsers(users) {
    // Array to store the pairs
    let pairs = [];
    
    // Loop over the users array for each user
    for (let i = 0; i < users.length; i++) {
        // Maximum average rating and the user who has the maximum average
        let maxAvg = Number.MIN_SAFE_INTEGER;
        let maxAvgUser = null;
        
        // Get the ratings of the current user's skills
        let user1Skills = users[i].skills;

        // Loop over the remaining users in the array
        for (let j = i+1; j < users.length; j++) {
            // Get the ratings of the current comparing user's skills
            let user2Skills = users[j].skills;

            // Initialize the total and count to calculate the average
            let total = 0;
            let count = 0;

            // Loop over the first user's skills
            for (let skill in user1Skills) {
                // If the second user also has the same skill
                if (skill in user2Skills) {
                    // Add to the total and increase the count
                    total += (user1Skills[skill] + user2Skills[skill]);
                    count += 2;
                }
            }
          
            // If there were common skills found
            if (count > 0) {
                // Calculate the average
                let avg = total / count;

                // If this average is greater than the maximum average found so far
                if (avg > maxAvg) {
                    // Set the maximum average and user
                    maxAvg = avg;
                    maxAvgUser = users[j];
                }
            }
        }

        // If a user was found
        if (maxAvgUser) {
            // Add the pair to the pairs array
            pairs.push([users[i].name, maxAvgUser.name]);
            // Remove the second user from the users array
            users.splice(users.indexOf(maxAvgUser), 1);
        }
    }

    // Return the pairs
    return pairs;
}
