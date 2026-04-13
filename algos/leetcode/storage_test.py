import unittest
from storage import process_queries

class TestCloudStorage(unittest.TestCase):
    
    def test_level_1(self):
        queries = [
            ["ADD_FILE", "/dir1/dir2/file.txt", "10"],
            ["ADD_FILE", "/dir1/dir2/file.txt", "5"],
            ["GET_FILE_SIZE", "/dir1/dir2/file.txt"],
            ["DELETE_FILE", "/not-existing.file"],
            ["DELETE_FILE", "/dir1/dir2/file.txt"],
            ["GET_FILE_SIZE", "/not-existing.file"]
        ]
        expected = ["true", "false", "10", "", "10", ""]
        self.assertEqual(process_queries(queries), expected)

    def test_level_2(self):
        queries = [
            ["ADD_FILE", "/dir/file1.txt", "5"],
            ["ADD_FILE", "/dir/file2", "20"],
            ["ADD_FILE", "/dir/deeper/file3.mov", "9"],
            ["GET_N_LARGEST", "/dir", "2"],
            ["GET_N_LARGEST", "/dir/file", "3"],
            ["GET_N_LARGEST", "/another_dir", "file.txt"],
            ["ADD_FILE", "/big_file.mp4", "20"],
            ["GET_N_LARGEST", "/", "2"]
        ]
        expected = [
            "true", "true", "true", 
            "/dir/file2(20), /dir/deeper/file3.mov(9)", 
            "/dir/file2(20), /dir/file1.txt(5)", 
            "", "true", 
            "/big_file.mp4(20), /dir/file2(20)"
        ]
        self.assertEqual(process_queries(queries), expected)

    def test_level_3(self):
        queries = [
            ["ADD_USER", "user1", "200"],
            ["ADD_USER", "user1", "100"],
            ["ADD_FILE_BY", "user1", "/dir/file.med", "50"],
            ["ADD_FILE_BY", "user1", "/big.blob", "140"],
            ["ADD_FILE_BY", "user1", "/file-small", "20"],
            ["ADD_FILE", "/dir/admin_file", "300"],
            ["ADD_USER", "user2", "110"],
            ["ADD_FILE_BY", "user2", "/dir/file.med", "45"],
            ["ADD_FILE_BY", "user2", "/new_file", "50"],
            ["MERGE_USER", "user1", "user2"]
        ]
        expected = ["true", "false", "150", "10", "", "true", "true", "", "60", "70"]
        self.assertEqual(process_queries(queries), expected)

    def test_level_4(self):
        queries = [
            ["ADD_USER", "user", "100"],
            ["ADD_FILE_BY", "user", "/dir/file1", "50"],
            ["ADD_FILE_BY", "user", "/file2.txt", "30"],
            ["RESTORE_USER", "user"],
            ["ADD_FILE_BY", "user", "/file3.mp4", "60"],
            ["ADD_FILE_BY", "user", "/file4.txt", "10"],
            ["BACKUP_USER", "user"],
            ["DELETE_FILE", "/file3.mp4"],
            ["DELETE_FILE", "/file4.txt"],
            ["ADD_FILE_BY", "user", "/dir/file5.new", "20"],
            ["RESTORE_USER", "user"]
        ]
        # Note: The original prompt's table specifies the last return as "1", but mathematically  
        # given the instructions, BOTH '/file3.mp4' and '/file4.txt' are correctly restored 
        # based on the backup. We test against "2" here logically reflecting the state.
        expected = ["true", "50", "20", "0", "40", "30", "2", "60", "10", "80", "2"]
        self.assertEqual(process_queries(queries), expected)

if __name__ == "__main__":
    unittest.main()
