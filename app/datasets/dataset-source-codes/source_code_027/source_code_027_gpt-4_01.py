def move_zeros_to_end(nums):
    # Counter for number of zeros in the list
    zero_count = 0

    # Traverse the given list
    for i in range(len(nums)):
        # If element encountered is non-zero, then replace the element at index 'count' 
        # with this element and increment 'count'
        if nums[i] != 0:
            nums[zero_count] = nums[i]
            zero_count += 1
            
    # Now all non-zero elements have been shifted to front and 'count' is set as index of first 0.  
    # Make all remaining elements as 0.
    while zero_count < len(nums):
        nums[zero_count] = 0
        zero_count += 1

    return nums

# Testing with the given examples
print(move_zeros_to_end([0, 1, 0, 3, 12]))  # Expected Output: [1, 3, 12, 0, 0]
print(move_zeros_to_end([0, 0, 1]))  # Expected Output: [1, 0, 0]
