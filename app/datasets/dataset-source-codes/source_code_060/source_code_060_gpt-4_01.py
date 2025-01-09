def merge_intervals(intervals):
    # Sort the intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Instantiate an empty list to hold merged intervals
    merged = []

    # Loop over the intervals
    for interval in intervals:
        # If the list of merged intervals is empty or if the current interval does not overlap with the previous, append it
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, merge the intervals
            merged[-1][1] = max(merged[-1][1], interval[1])

    # Return the merged intervals
    return merged
