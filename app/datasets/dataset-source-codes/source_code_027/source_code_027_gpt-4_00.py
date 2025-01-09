def rearrange(nums):
    # Index to insert a non-zero element
    insert_index = 0

    # Iterate over the list
    for i in range(len(nums)):
        # If the element is not zero, insert it at the insert_index position
        # Then increment the insert_index
        if nums[i] != 0:
            nums[insert_index] = nums[i]
            insert_index += 1

    # At this point, all non-zero elements have been moved to the beginning of the list
    # in their original order. Now we just need to fill in the rest of the list with zeros.
    for i in range(insert_index, len(nums)):
        nums[i] = 0

    return nums

# Testing the function
print(rearrange([0, 1, 0, 3, 12]))  # Output: [1, 3, 12, 0, 0]
print(rearrange([0, 0, 1]))  # Output: [1, 0, 0]
