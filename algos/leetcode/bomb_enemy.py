"""
Given an m x n matrix grid where each cell is either a wall 'W', an enemy 'E' or empty '0', return the maximum enemies you can kill using one bomb. You can only place the bomb in an empty cell.

The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since it is too strong to be destroyed.

Input: grid = [["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]
Output: 3
"""

class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        R = len(grid)
        C = len(grid[0])
        def count_kills(r,c):
            count = 0
            # curr block
            if grid[r][c] == 'E':
                count +=1
            # left side of row
            for c_i in range(c-1, -1, -1):
                if grid[r][c_i] == 'W':
                    break
                elif grid[r][c_i] == 'E':
                    count += 1
            #right side of row
            for c_i in range(c+1, C):
                if grid[r][c_i] == 'W':
                    break
                elif grid[r][c_i] == 'E':
                    count += 1
            
            # top side of column
            for r_i in range(r-1, -1, -1):
                if grid[r_i][c] == 'W':
                    break
                elif grid[r_i][c] == 'E':
                    count += 1
            #bottom side of column
            for r_i in range(r+1, R):
                if grid[r_i][c] == 'W':
                    break
                elif grid[r_i][c] == 'E':
                    count += 1
            print(count, r,c)
            return count

        result = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '0':
                    result = max(result, count_kills(r, c))
        return result
    
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        if len(grid) == 0:
            return 0

        rows, cols = len(grid), len(grid[0])

        max_count = 0
        row_hits = 0
        col_hits = [0] * cols

        for row in range(0, rows):
            for col in range(0, cols):
                # reset the hits on the row, if necessary.
                if col == 0 or grid[row][col - 1] == 'W':
                    row_hits = 0
                    for k in range(col, cols):
                        if grid[row][k] == 'W':
                            # stop the scan when we hit the wall.
                            break
                        elif grid[row][k] == 'E':
                            row_hits += 1

                # reset the hits on the col, if necessary.
                if row == 0 or grid[row - 1][col] == 'W':
                    col_hits[col] = 0
                    for k in range(row, rows):
                        if grid[k][col] == 'W':
                            break
                        elif grid[k][col] == 'E':
                            col_hits[col] += 1

                # count the hits for each empty cell.
                if grid[row][col] == '0':
                    total_hits = row_hits + col_hits[col]
                    max_count = max(max_count, total_hits)

        return max_count
