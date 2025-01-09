def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)

    return merged

# Input example
input_intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print("Input:", input_intervals)
print("Output:", merge_intervals(input_intervals))

# Another input example
input_intervals = [[1, 4], [4, 5]]
print("\nInput:", input_intervals)
print("Output:", merge_intervals(input_intervals))
