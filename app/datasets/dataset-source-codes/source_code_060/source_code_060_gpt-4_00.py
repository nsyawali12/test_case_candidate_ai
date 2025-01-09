def merge_intervals(intervals):
    # Sorting the intervals based on their start times
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        # If the list of merged intervals is empty, or if the current
        # interval does not overlap with the previous, append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is an overlap, so we merge the current and previous
            # intervals by updating the end time of the previous interval.
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged

# Test the function with given examples
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]])) # Expected: [[1,6],[8,10],[15,18]]
print(merge_intervals([[1,4],[4,5]])) # Expected: [[1,5]]
