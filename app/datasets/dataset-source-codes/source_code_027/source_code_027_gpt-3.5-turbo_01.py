def move_zeros_to_end(nums):
    count = 0
    n = len(nums)

    for i in range(n):
        if nums[i] != 0:
            nums[i], nums[count] = nums[count], nums[i]
            count += 1

# Test cases
nums1 = [0, 1, 0, 3, 12]
move_zeros_to_end(nums1)
print(nums1)  # Output: [1, 3, 12, 0, 0]

nums2 = [0, 0, 1]
move_zeros_to_end(nums2)
print(nums2)  # Output: [1, 0, 0]

