import os

from image import ImageUtils
from loader import ImageLoader


class DataPreparator:

    WIDTH = 128
    HEIGHT = 128

    @staticmethod
    def prepare_data(path):
        loader = ImageLoader()
        imagesData = loader.load(path)

        images = []
        breeds = []
        for data in imagesData:
            for obj in data.objects:
                if data.data:
                    image = ImageUtils.crop(data.data, obj.minX, obj.minY, obj.maxX, obj.minY)
                    image = ImageUtils.resize(image, DataPreparator.WIDTH, DataPreparator.HEIGHT)
                    # TODO: tutaj jakoś pobrać dane
                    images.append(image.getdata())
                    breeds.append(obj.name)

        return images, breeds