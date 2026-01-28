class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = {}
        outdegree = {}
        have_req = set()
        for req in prerequisites:
            child, parent = req[0], req[1]
            have_req.add(child)
            if parent not in outdegree:
                outdegree[parent] = [child]
            else:
                outdegree[parent].append(child)
            if child not in indegree:
                indegree[child] = 1 
            else:
                indegree[child] += 1
        order = []
        no_req = set(i for i in range(numCourses)).difference(have_req)
        q = deque(no_req)
        while(q):
            course = q.popleft()
            order.append(course)
            if course in outdegree:
                for child in outdegree[course]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        q.append(child)
        if len(order) != numCourses:
            return []
        return order

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = defaultdict(int)
        successors = defaultdict(list)
        have_req = set()
        schedule = []
        for prereq in prerequisites:
            course, pre = prereq[0], prereq[1]
            indegree[course] += 1
            successors[pre].append(course)
            have_req.add(course)
        q = deque()
        for no_prereq in {_ for _ in range(numCourses)} - have_req:
            q.append(no_prereq)
        while(q):
            course = q.popleft()
            schedule.append(course)
            for next_course in successors[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    q.append(next_course)
        if len(schedule) != numCourses:
            return []
        else:
            return schedule

