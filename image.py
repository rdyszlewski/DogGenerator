from PIL import Image


class ImageUtils:

    @staticmethod
    def crop(image, xmin, ymin, xmax, ymax):
        return image.crop((xmin, ymin, xmax, ymax))

    @staticmethod
    def save(image, path):
        image.save(path)

    @staticmethod
    def resize(image, x, y):
        return image.resize((x, y), Image.ANTIALIAS)