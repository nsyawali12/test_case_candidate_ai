def move_zeros_to_end(nums):
    # Index to place the next non-zero element
    position = 0

    # First pass to handle all non-zero elements
    for num in nums:
        if num != 0:
            nums[position] = num
            position += 1

    # Fill the rest of the array with zeros
    for i in range(position, len(nums)):
        nums[i] = 0

# Example usage:
nums1 = [0, 1, 0, 3, 12]
move_zeros_to_end(nums1)
print(nums1)  # Output: [1, 3, 12, 0, 0]

nums2 = [0, 0, 1]
move_zeros_to_end(nums2)
print(nums2)  # Output: [1, 0, 0]
