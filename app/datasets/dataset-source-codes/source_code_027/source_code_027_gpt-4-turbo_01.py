def move_zeros_to_end(nums):
    # Initialize an index to keep track of the position to place the non-zero element
    insert_pos = 0
    
    # Loop through each element in the list
    for num in nums:
        # If the element is not zero, place it at the current insert position
        # and move the insert position forward
        if num != 0:
            nums[insert_pos] = num
            insert_pos += 1
    
    # After all non-zero elements are relocated, fill the remaining positions with zeroes
    while insert_pos < len(nums):
        nums[insert_pos] = 0
        insert_pos += 1
