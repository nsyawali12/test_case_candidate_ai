def evenSum(numList):
  if len(numList) == 0:
    return 0
  return sum(num for num in numList if num % 2 == 0)