import math

class InMemoryDB:
    def __init__(self):
        # Main db storage: {key: {field: (value, expiration_timestamp)}}
        self.db = {}
        # Backup storage: [(backup_timestamp, snapshot_dict)]
        self.backups = []

    def _is_alive(self, key: str, field: str, current_time: int = None) -> bool:
        if key not in self.db or field not in self.db[key]:
            return False
        if current_time is None:
            return True
        _, expire_time = self.db[key][field]
        return current_time < expire_time

    # --- Level 1 ---
    
    def set(self, key: str, field: str, value: str) -> str:
        if key not in self.db:
            self.db[key] = {}
        self.db[key][field] = (value, math.inf)
        return ""

    def get(self, key: str, field: str) -> str:
        if self._is_alive(key, field):
            return self.db[key][field][0]
        return ""

    def delete(self, key: str, field: str) -> str:
        if self._is_alive(key, field):
            del self.db[key][field]
            if not self.db[key]:
                del self.db[key]
            return "true"
        return "false"

    # --- Level 2 ---

    def scan(self, key: str) -> str:
        if key not in self.db:
            return ""
        fields = [f"{f}({v})" for f, (v, _) in self.db[key].items()]
        return ", ".join(sorted(fields))

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        if key not in self.db:
            return ""
        fields = [f"{f}({v})" for f, (v, _) in self.db[key].items() if f.startswith(prefix)]
        return ", ".join(sorted(fields))

    # --- Level 3 ---

    def set_at(self, key: str, field: str, value: str, timestamp: str) -> str:
        if key not in self.db:
            self.db[key] = {}
        self.db[key][field] = (value, math.inf)
        return ""

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: str, ttl: str) -> str:
        ts, t = int(timestamp), int(ttl)
        if key not in self.db:
            self.db[key] = {}
        self.db[key][field] = (value, ts + t)
        return ""

    def delete_at(self, key: str, field: str, timestamp: str) -> str:
        ts = int(timestamp)
        if self._is_alive(key, field, ts):
            del self.db[key][field]
            if not self.db[key]:
                del self.db[key]
            return "true"
        return "false"

    def get_at(self, key: str, field: str, timestamp: str) -> str:
        ts = int(timestamp)
        if self._is_alive(key, field, ts):
            return self.db[key][field][0]
        return ""

    def scan_at(self, key: str, timestamp: str) -> str:
        ts = int(timestamp)
        if key not in self.db:
            return ""
        fields = [f"{f}({v})" for f, (v, exp) in self.db[key].items() if ts < exp]
        return ", ".join(sorted(fields))

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: str) -> str:
        ts = int(timestamp)
        if key not in self.db:
            return ""
        fields = [f"{f}({v})" for f, (v, exp) in self.db[key].items() if f.startswith(prefix) and ts < exp]
        return ", ".join(sorted(fields))

    # --- Level 4 ---

    def backup(self, timestamp: str) -> str:
        ts = int(timestamp)
        snapshot = {}
        non_empty_keys = 0
        
        for k, fields in self.db.items():
            record_snapshot = {}
            for f, (v, exp) in fields.items():
                if ts < exp:
                    # Calculate remaining lifespan relative to the backup timestamp
                    rem_ttl = exp - ts if exp != math.inf else math.inf
                    record_snapshot[f] = (v, rem_ttl)
            if record_snapshot:
                snapshot[k] = record_snapshot
                non_empty_keys += 1
                
        self.backups.append((ts, snapshot))
        return str(non_empty_keys)

    def restore(self, timestamp: str, timestampToRestore: str) -> str:
        ts, target_ts = int(timestamp), int(timestampToRestore)
        
        target_snapshot = None
        # Backups are guaranteed to be chronological; find the latest one before timestampToRestore
        for b_ts, snap in reversed(self.backups):
            if b_ts < target_ts:
                target_snapshot = snap
                break
                
        if target_snapshot is not None:
            self.db = {}
            for k, fields in target_snapshot.items():
                self.db[k] = {}
                for f, (v, rem_ttl) in fields.items():
                    # Recalculate future expiration based on current restore timestamp
                    new_exp = ts + rem_ttl if rem_ttl != math.inf else math.inf
                    self.db[k][f] = (v, new_exp)
                    
        return ""
