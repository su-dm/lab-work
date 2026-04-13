
class Cloud:
    def __init__(self):
        admin = {'cap':float('inf'), 'files': set(), 'usage': 0, 'backup': None}
        self.users = {'admin':admin}
        self.files = {}

    def add_file(self, name, size):
        size = int(size)
        if name in self.files:
            return "false"
        self.files[name] = (size, 'admin')
        self.users['admin']['files'].add(name)
        self.users['admin']['usage'] += size
        return "true"

    def get_file_size(self, name):
        if name not in self.files:
            return ""
        return str(self.files[name][0])

    def delete_file(self, name):
        if name not in self.files:
            return ""
        file_size, user = self.files[name]
        del self.files[name]
        self.users[user]['files'].remove(name)
        self.users[user]['usage'] -= file_size
        return str(file_size)

    def get_n_largest(self, prefix, n):
        try:
            n = int(n)
        except ValueError:
            return ""
        filtered = [i for i in self.files.items() if i[0].startswith(prefix)]
        top = sorted(filtered, key=lambda x: (-x[1][0],x[0]))[:n]
        if not top:
            return ""
        return ", ".join(f"{x[0]}({x[1][0]})" for x in top)

    def add_user(self, user_id, capacity):
        capacity = int(capacity)
        if user_id in self.users:
            return "false"
        self.users[user_id] = {'cap':capacity, 'files':set(), 'usage':0,
                               'backup':None}
        return "true"

    def add_file_by(self, user_id, name, size):
        size = int(size)
        if user_id not in self.users:
            return ""
        if name in self.files:
            return ""

        user = self.users[user_id]
        if size > user['cap'] - user['usage']:
            return ""
        user['files'].add(name)
        user['usage']+= size
        self.files[name] = (size, user_id)
        return str(user['cap'] - user['usage'])

    def merge_user(self, user_id1, user_id2):
        if user_id1 not in self.users or user_id2 not in self.users or user_id1 == user_id2:
            return ""

        user1 = self.users[user_id1]
        user2 = self.users[user_id2]
        u2files = user2['files']
        user1['cap'] += user2['cap']
        user1['usage'] += user2['usage']
        u2files = user2['files']
        for fname in u2files:
            f = self.files[fname]
            self.files[fname] = (f[0], user_id1)
            user1['files'].add(fname)
        del self.users[user_id2]
        return str(user1['cap'] - user1['usage'])

    def backup_user(self, user_id):
        if user_id not in self.users:
            return ""
        user = self.users[user_id]
        files = []
        for fname in user['files']:
            file_size, _ = self.files[fname]
            files.append((fname, file_size))
        user['backup'] = files
        return str(len(files))

    def restore_user(self, user_id):
        if user_id not in self.users:
            return ""
        user = self.users[user_id]
        if user.get('backup', None) is None:
            # delete all files
            for fname in user['files']:
                del self.files[fname]
            user['usage'] = 0
            user['files'] = set()
            return '0'
        backup = user['backup']
        restored = 0
        usage = 0
        user_r = {'cap': user['cap'], 'files':set(), 'usage':0, 'backup':backup}
        #for f in backup:
        for name, size in backup:
            #name, size = f
            if name in self.files:
                e_size, e_user = self.files[name]
                if e_user != user_id:
                    continue
                if e_size != size:
                    self.files[name] = (size, user_id)
                    usage += size
                    restored+=1
            else:
                self.files[name] = (size, user_id)
                usage += size
                user_r['files'].add(name)
                restored+=1

        user_r['usage'] = usage
        self.users[user_id]=user_r
        return str(restored)

def process_queries(queries: list[list[str]]) -> list[str]:
    storage = Cloud()
    results = []
    
    for query in queries:
        op = query[0]
        if op == "ADD_FILE":
            results.append(storage.add_file(query[1], query[2]))
        elif op == "GET_FILE_SIZE":
            results.append(storage.get_file_size(query[1]))
        elif op == "DELETE_FILE":
            results.append(storage.delete_file(query[1]))
        elif op == "GET_N_LARGEST":
            results.append(storage.get_n_largest(query[1], query[2]))
        elif op == "ADD_USER":
            results.append(storage.add_user(query[1], query[2]))
        elif op == "ADD_FILE_BY":
            results.append(storage.add_file_by(query[1], query[2], query[3]))
        elif op == "MERGE_USER":
            results.append(storage.merge_user(query[1], query[2]))
        elif op == "BACKUP_USER":
            results.append(storage.backup_user(query[1]))
        elif op == "RESTORE_USER":
            results.append(storage.restore_user(query[1]))
            
    return results
