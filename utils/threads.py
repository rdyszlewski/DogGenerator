import os
import sys


class ThreadsUtils:

    @staticmethod
    def get_available_threads():
        if sys.platform == 'win32':
            return (int)(os.environ['NUMBER_OF_PROCESSORS'])
        else:
            return (int)(os.popen('grep -c cores /proc/cpuinfo').read())