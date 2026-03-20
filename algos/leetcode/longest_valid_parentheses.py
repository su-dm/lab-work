class Solution:
    """
    Use a stack to track the indices of valid parentheses. The thing is you track the idx before.
    So it's like right-left+1 equivalent you're doing right-(left-1)
    Key is to remember to start with pushing -1, cause if your 0 index is valid the one before that is -1
    You keep pushing openings and pop when you see closes. Then if there's something left on the stack that's the one right before the beginning of your valid one. You  can check for max.
    If there's no one before or anything on the stack left then you know it's invalid so you push your current index to keep track of "right before the start of a valid".
    If the next idx isn't valid it's fine this one gets popped and the next one gets pushed and we keep moving along until we find another valid.
    
    """
    def longestValidParentheses(self, s: str) -> int:
        maxans = 0
        stack = []
        stack.append(-1)
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxans = max(maxans, i - stack[-1])
        return maxans


    """ 
        In this solution do two passes left to right and right to left.
        Left to right you count to make sure theres more openings than closes, soon as r > l it's invalid another opening can't open a close that's behind it. Reset your left,right open/close counters and keep going.
        Similarly right to left you identify a break when l>r and reset your counters. If l==r check the max length.
        (((()
        ())))
        This is why we need both passes. If I'm focusing on openings and it's not invalid yet then I'm not checking length.
    """
    def longestValidParentheses(self, s: str) -> int:
        l, r, mx_len = 0, 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                l += 1
            else:
                r += 1
            if l == r:
                mx_len = max(mx_len, l*2)
            elif r > l:
                l,r = 0,0
        l, r = 0, 0
        for i in range(len(s)-1, -1, -1):
            if s[i] == '(':
                l += 1
            else:
                r += 1
            if l == r:
                mx_len = max(mx_len, l*2)
            elif l > r:
                l,r = 0,0
        return mx_len
