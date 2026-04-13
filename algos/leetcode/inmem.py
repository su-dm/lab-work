from collections import defaultdict

class InMemoryDatabase:
    def __init__(self):
        self.store = defaultdict(dict)
        self.backups = []
    
    # ========== Level 1 Operations ==========
    def set(self,key, field, value):
        self.store[key][field] = {'val':value}
        return "" 
    
    def get(self,key, field):
        db = self.store[key]
        if field not in db:
            return ""
        else:
            return db[field]['val']

    def delete(self,key, field):
        db = self.store[key]
        if field not in db:
            return False
        else:
            del db[field]
            return True
    
    # ========== Level 2 Operations ==========
    def scan(self, key):
        if key not in self.store:
            return ""
        return ", ".join(f'{x[0]}({x[1]['val']})' for x in
                        sorted(self.store[key].items()))


    def scan_by_prefix(self, key, prefix):
        return ", ".join(f'{x[0]}({x[1]['val']})' for x in
                        sorted(self.store[key].items()) if
                         x[0].startswith(prefix))

    # ========== Level 3 Operations ==========
    def set_at(self, key, field, value, timestamp):
        db = self.store[key]
        db[field] = {'val': value, 'ts': timestamp}
        return ""

    def set_at_with_ttl(self, key, field, value, timestamp, ttl):
        db = self.store[key]
        db[field] = {'val': value, 'ts': timestamp, 'ttl': timestamp+ttl}
        return ""

    def delete_at(self, key, field, timestamp):
        db = self.store[key]
        if field in db:
            f = db[field]
            if 'ttl' in f and timestamp >= f['ttl']:
                del db[field]
                return False
            del db[field]   
            return True
        return False

    def get_at(self, key, field, timestamp):
        db = self.store[key]
        if field not in db:
            return ""   
        f = db[field]
        if 'ttl' in f and timestamp >= f['ttl']:
            return ""
        return f["val"]

    def scan_at(self, key, timestamp):
        items = self.store[key].items()

        def filter_not_expired(x):
            return not ('ttl' in x and x['ttl'] <= timestamp)
        items = filter(lambda x: filter_not_expired(x[1]), items) 
        return ", ".join(f'{x[0]}({x[1]['val']})' for x in
                        sorted(items))

    def scan_by_prefix_at(self, key, prefix, timestamp):
        items = self.store[key].items()
        def filter_not_expired(x):
            return not ('ttl' in x and x['ttl'] <= timestamp)
        items = filter(lambda x: filter_not_expired(x[1]) 
                       and x[0].startswith(prefix), items)
        return ", ".join(f'{x[0]}({x[1]['val']})' for x in
                        sorted(items))

    # ========== Level 4 Operations ==========
    def backup(self, timestamp):
        import copy
        b = copy.deepcopy(self.store)
        self.backups.append((timestamp, b))
        count = 0
        for k,v in b.items():
            for f,e in v.items():
                if 'ttl' not in e or e['ttl'] > timestamp:
                    count += 1
                    break
        return str(count)
        
    def restore(self, timestamp, timestampToRestore):
        import copy
        t, b = None, None
        for i in range(len(self.backups) - 1):
            t1, b1 = self.backups[i]
            t2, b2 = self.backups[i+1]
            if t2 > timestampToRestore:
                t, b = t1, b1
                break
        if b is None:
            t, b = self.backups[-1]
        # t is when it was backed up and frozen
        # field: {'ts', 'ttl'} so timestamp + (ttl - t) is new ttl
        new_store = copy.deepcopy(b)
        for key, db in new_store.items():
            for f,e in db.items():
                if 'ttl' in e:
                    e['ttl'] = timestamp + (e['ttl'] - t)
        self.store = new_store
        return ""
