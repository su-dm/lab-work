"Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area."

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        R = len(matrix)
        C = len(matrix[0])
        max_area = 0
        # width of the row starting from column
        wrc = [[0] * (C+1) for _ in range(R)]
        for r in range(R):
            for c in range(1,C+1):
                # 1 index wrc to be able to look back for first element, -1 to 0 index on matrix
                wrc[r][c] = wrc[r][c-1] + 1 if matrix[r][c-1] == '1' else 0
                
                # calc area going up
                w = wrc[r][c]
                for top_r in range(r, -1, -1):
                    w = min(w, wrc[top_r][c])
                    area = w * (r - top_r + 1)
                    max_area = max(max_area, area)
        return max_area

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        maxarea = 0

        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "0":
                    continue

                # compute the maximum width and update dp with it
                width = dp[i][j] = dp[i][j - 1] + 1 if j else 1

                # compute the maximum area rectangle with a lower right corner at [i, j]
                for k in range(i, -1, -1):
                    width = min(width, dp[k][j])
                    maxarea = max(maxarea, width * (i - k + 1))
        return maxarea
        
        

                    
