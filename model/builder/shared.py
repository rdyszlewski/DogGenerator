import re

class LayersPattern:

    @staticmethod
    def getParameters(value):
        pattern = '\((.*)\)'
        value = re.search(pattern, value).group(1)
        return value