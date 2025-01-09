import random

def rolling():
  rand_num = random.choice([1,2,3,4,5,6])
  return rand_num

def prob(int(n)):
  occurrence = dict()
  for i in len(n):
    num = rolling()
    if num == 1:
      count_1 = count_1 + 1
    else if num == 2:
      count_2 = count_2 + 1 
    else if num == 3:
      count_3 = count_3 + 1
    else if num == 4:
      count_4 = count_4 + 1
    else if num == 5:
      count_5 = count_5 + 1
    else if num == 6:
      count_6 = count_6 + 1

    occurrence["count_1"] = count_1
    occurrence["count_2"] = count_2
    occurrence["count_3"] = count_3
    occurrence["count_4"] = count_4
    occurrence["count_5"] = count_5
    occurrence["count_6"] = count_6
    
    return occurrence