class DogImage:

    def __init__(self):
        self.name=''
        self.width = 0
        self.height = 0
        self.data = None
        self.objects = []


class ImageObject:

    def __init__(self):
        self.name = ''
        self.pose = ''
        self.truncated = False

        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0