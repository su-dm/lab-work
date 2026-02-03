def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    i = 1
    while(i < len(intervals)):
        if intervals[i][0] <= intervals[i-1][1]:
            intervals[i-1][1] = max(intervals[i][1], intervals[i-1][1])
            intervals.pop(i)
        else:
            i += 1
    return intervals

# pop(i) is slow, O(n), overwrite with pointers and truncate at the end to do in-place optimally
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    write = 0
    for read in range(1, len(intervals)):
        if intervals[read][0] <= intervals[write][1]:
            intervals[write][1] = max(intervals[write][1], intervals[read][1])
        else:
            write += 1
            intervals[write] = intervals[read]
    return intervals[:write + 1]
