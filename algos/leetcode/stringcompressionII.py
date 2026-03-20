"""
Run-length encoding is a string compression method that works by replacing consecutive identical characters (repeated 2 or more times) with the concatenation of the character and the number marking the count of the characters (length of the run). For example, to compress the string "aabccc" we replace "aa" by "a2" and replace "ccc" by "c3". Thus the compressed string becomes "a2bc3".

Notice that in this problem, we are not adding '1' after single characters.

Given a string s and an integer k. You need to delete at most k characters from s such that the run-length encoded version of s has minimum length.

Find the minimum length of the run-length encoded version of s after deleting at most k characters.
"""



class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        N = len(s)
        @lru_cache
        def dp(idx, last_char, last_char_count, k):
            if idx == N:
                #print(idx, last_char, last_char_count, k, "ZERO")
                return 0
            # don't delete
            dont_delete = float('inf')
            char_left = N - (idx + 1)
            # need k remaining characters to delete
            if char_left >= k:
                if s[idx] == last_char:
                    if last_char_count in {1, 9, 99}:
                        dont_delete = dp(idx+1, last_char, last_char_count+1, k) + 1
                    else:
                        dont_delete = dp(idx+1, last_char, last_char_count+1, k)
                else:
                    dont_delete = dp(idx+1, s[idx], 1, k) + 1

            # delete
            delete = float('inf')
            if k > 0:
                delete = dp(idx+1, last_char, last_char_count, k-1)
            #print(idx, last_char, last_char_count, k, delete, dont_delete)
            return min(delete, dont_delete)
        res = dp(0, '', 0, k)
        print(res)
        return res

    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        @lru_cache(None)
        def dp(idx, last_char, last_char_count, k):
            if k < 0:
                return float('inf')
            if idx == n:
                return 0
            
            delete_char = dp(idx + 1, last_char, last_char_count, k - 1)
            if s[idx] == last_char:
                keep_char = dp(idx + 1, last_char, last_char_count + 1, k) + (last_char_count in [1, 9, 99])
            else:
                keep_char = dp(idx + 1, s[idx], 1, k) + 1
            
            return min(keep_char, delete_char)
        
        n = len(s)
        return dp(0, "", 0, k)
