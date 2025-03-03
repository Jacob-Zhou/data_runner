
from functools import partial
from io import TextIOWrapper
import json
from typing import Dict, Any, Set
from threading import Lock


class ThreadSafeFileWriter():
    """
    A task that writes data to a file, console, or error stream.
    
    This task is thread-safe and manages file handles to ensure proper resource usage.
    
    Attributes:
        file_path (str): The path to the file to write to.
    """
    
    # thread safe file writing task, hold the file handle in the class instance
    # use a lock to ensure thread safe
    __file_handle: Dict[str, TextIOWrapper] = {}
    __file_usage_count: Dict[str, int] = {}
    __file_opened: Set[str] = set()
    __lock: Lock = Lock()

    def __init__(self, file_path: str = None):
        """
        Initialize a file writing task.
        
        Args:
            signature (str): The signature defining input and output fields.
            file_path (str): The path to the file to write to.
        """
        self.file_path = file_path
        assert file_path is not None, "file_path is required"
        with self.__lock:
            if file_path not in self.__file_handle:
                if file_path not in self.__file_opened:
                    self.__file_opened.add(file_path)
                    flag = 'w'
                else:
                    flag = 'a'
                self.__file_handle[file_path] = open(file_path, flag)
                # print(f"Opened file {file_path}, flag: {flag}")
                self.__file_usage_count[file_path] = 0
            self.__file_usage_count[file_path] += 1

    def write(self, data: str):
        """
        Write the input data to the specified output destination.
        
        Args:
            kwargs (Dict[str, Any]): The data to write.
            
        Returns:
            Dict[str, Any]: The output data (typically empty).
        """
        with self.__lock:
            self.__file_handle[self.file_path].write(data)

    def flush(self):
        """
        Flush the file handle.
        """
        with self.__lock:
            self.__file_handle[self.file_path].flush()

    def __del__(self):
        """
        Clean up file handles when the task is deleted.
        """
        with self.__lock:
            self.__file_usage_count[self.file_path] -= 1
            if self.__file_usage_count[self.file_path] == 0:
                self.__file_handle[self.file_path].close()
                del self.__file_handle[self.file_path]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

def write_to_file(file_path: str, data: str):
    """
    Thread safe file writing function.
    """
    with ThreadSafeFileWriter(file_path) as file_writer:
        file_writer.write(data)

def to_file_writer_error_handler(file_path: str, error_message: str, context: Dict[str, Any]):
    """
    Error handler that writes to a file.
    """
    with ThreadSafeFileWriter(file_path) as file_writer:
        file_writer.write(json.dumps({
            "error_message": error_message,
            "context": context
        }, ensure_ascii=False) + "\n")
        file_writer.flush()

def get_to_file_writer_error_handler(file_path: str):
    """
    Get a function that writes to a file.
    """
    return partial(to_file_writer_error_handler, file_path)
