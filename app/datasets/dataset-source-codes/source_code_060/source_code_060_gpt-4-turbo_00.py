def merge_intervals(intervals):
    # Sort the list of intervals by their start values
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    
    for interval in intervals:
        # If the list of merged intervals is empty or if the current interval does not overlap with the previous one
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # There is overlap, merge the current interval with the previous one
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged
