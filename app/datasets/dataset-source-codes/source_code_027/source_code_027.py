def move_zeroes(nums):
  count = 0
  for i in range(len(nums)):
    if nums[i] != 0:
      nums[count], nums[i]= nums[i], nums[count]
      count += 1
  for i in range(count, len(nums)):
    nums[i] =0

input = [int(x) for x in input("Enter integers separated by spaces: ").split()]
move_zeroes(input)

print(input)