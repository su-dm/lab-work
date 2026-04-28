"""
FILE_UPLOAD(file_name, size)
Upload the file to the remote storage server.
If a file with the same name already exists on the server, it throws a runtime exception.
FILE_GET(file_name)
Returns the size of the file, or nothing if the file doesn’t exist.
FILE_COPY(source, dest)
Copy the source file to a new location.
If the source file doesn’t exist, it throws a runtime exception.
If the destination file already exists, it overwrites the existing file.

LEVEL 2
FILE_SEARCH(prefix)
Find top 10 files starting with the provided prefix. 
Order results by their size in descending order, and in case of a tie by file name.

LEVEL 3
Files now might have a specified time to live on the server. 
Implement extensions of existing methods which inherit all functionality 
but also with an additional parameter to include a timestamp for the operation,
and new files might specify the time to live - no ttl means lifetime being infinite.

FILE_UPLOAD_AT(timestamp, file_name, file_size)
FILE_UPLOAD_AT(timestamp, file_name, file_size, ttl)
The uploaded file is available for ttl seconds.
FILE_GET_AT(timestamp, file_name)
FILE_COPY_AT(timestamp, file_from, file_to)
FILE_SEARCH_AT(timestamp, prefix)
Results should only include files that are still “alive”.

LEVEL 4
ROLLBACK(timestamp)
Rollback the state of the file storage to the state specified in the timestamp.
All ttls should be recalculated accordingly.
"""

import copy
class File:
    def __init__(self, name, size, timestamp=None, ttl=None):
        self.timestamp = timestamp 
        self.name = name
        self.size = size
        self.ttl = ttl

class FileStore:
    def __init__(self):
        self.files = {}
        self.backups = {}

    # basic, TOOO: could do file history, query log reconstruction is too much
    def backup(self, timestamp):
        self.backups[timestamp] = copy.deepcopy(self.files)

    def restore(self, timestamp_backup, timestamp_now):
        if timestamp_backup not in self.backups:
            raise RuntimeError("Backup does not exist for that timestamp")
        b = self.backups[timestamp_backup]
        new_store = {}
        def adjust_timestamp(f):
            if not f.ttl:
                return f
            time_lived = timestamp_backup - f.timestamp
            time_left = f.ttl - time_lived
            if time_left:
                # test this
                new_file = copy.copy(f)
                new_file.ttl = timestamp_now + time_left
                return new_file
            return None
        for f in b.values():
            f_new = adjust_timestamp(f)
            if f_new:
                new_store[f_new.name] = f_new
        self.files = new_store

    def check_file(self, timestamp, file_name):
        file = self.files.get(file_name, None)
        if not file or (file.ttl and file.ttl + file.timestamp <= timestamp):
            return None
        return file

    def file_upload_at(self, timestamp, file_name, file_size, ttl=None):
        if self.check_file(timestamp, file_name):
            raise RuntimeError("File already exists")
        new_file = File(file_name, file_size, timestamp, ttl)
        self.files[file_name] = new_file
        return file_size

    def file_get_at(self, timestamp, file_name):
        if self.check_file(timestamp, file_name):
            return self.files[file_name]
        return None

    def file_copy_at(self, timestamp, source, dest):
        if not self.check_file(timestamp, source):
            raise RuntimeError("File doesn't exists")
        # test this
        new_file = copy.copy(self.files[source])
        new_file.name = dest
        # unclear
        new_file.timestamp = timestamp
        self.files[dest] = new_file

    # binary search, trie, bucket by letter
    def file_search_at(self, timestamp, prefix):
        # TODO: for now, simple
        matches = [(f.size, f.name) for f in self.files.values() if f.name.startswith(prefix) and self.check_file(timestamp, f.name)]
        return sorted(matches, key=lambda x: (-x[0], x[1]))[:10]

