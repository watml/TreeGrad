import os


class os_lock:
    def __init__(self, file):
        path_components = file.split(os.sep)
        path = os.sep.join(path_components[:-1])
        os.makedirs(path, exist_ok=True)
        self.lockfile = file + ".lock"
        self.lock_state = False
        self.file = file

    def acquire(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL)
        except OSError:
            return False
        else:
            os.close(fd)
            return True

    def release(self):
        os.remove(self.lockfile)

    def __enter__(self):
        if not os.path.exists(self.file):
            self.lock_state = self.acquire()
        return self.lock_state

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_state:
            self.release()