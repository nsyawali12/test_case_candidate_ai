function findBestPairings(users) {
  const pairs = []

  while (users.length > 1) {
    let bestPair = null
    let bestAverageRating = -1

    for (let i = 0; i < users.length - 1; i++) {
      for (let j = i + 1; j < users.length; j++) {
        const commonSkills = getCommonSkills(users[i].skills, users[j].skills)

        if (commonSkills.length > 0) {
          const averageRating = calculateAverageRating(
            commonSkills,
            users[i],
            users[j]
          )

          if (averageRating > bestAverageRating) {
            bestAverageRating = averageRating
            bestPair = [users[i].name, users[j].name]
          }
        }
      }
    }

    if (bestPair) {
      pairs.push(bestPair)
      users = users.filter(
        (user) => user.name !== bestPair[0] && user.name !== bestPair[1]
      )
    } else {
      break
    }
  }

  return pairs
}

function getCommonSkills(skills1, skills2) {
  return Object.keys(skills1).filter((skill) => skills2.hasOwnProperty(skill))
}

function calculateAverageRating(commonSkills, user1, user2) {
  const totalRating = commonSkills.reduce((sum, skill) => {
    return sum + user1.skills[skill] + user2.skills[skill]
  }, 0)

  return totalRating / commonSkills.length
}

const users = [
  { name: 'Alice', skills: { Java: 5, C: 3 } },
  { name: 'Bob', skills: { Java: 6, Python: 4 } },
  { name: 'Charlie', skills: { Python: 5, C: 5 } }
]

const pairings = findBestPairings(users)
console.log(pairings)