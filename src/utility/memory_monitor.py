import psutil
class MemoryMonitor:
    def __init__(self):
        self.max_memory = 0

    def update(self):
        # Get the current memory usage in bytes
        current_memory = psutil.Process().memory_info().rss
        if current_memory > self.max_memory:
            self.max_memory = current_memory

    def get_max_memory_allocated(self):
        return self.max_memory / (1024 ** 2)