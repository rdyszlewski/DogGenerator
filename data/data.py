import gc

import numpy

from image import ImageUtils
from loader import ImageLoader
from utils.encoding import OneHotEncoder


class DataPreparator:

    @staticmethod
    def prepare_data(path, width, height):
        loader = ImageLoader()
        imagesData = loader.load(path)

        images = []
        breeds = []
        for data in imagesData:
            for obj in data.objects:
                if data.data:
                    image = ImageUtils.crop(data.data, obj.minX, obj.minY, obj.maxX, obj.maxY)
                    image = ImageUtils.resize(image, width, height)
                    image_array = numpy.array(image)
                    del image
                    # TODO przerobić to jako tablica
                    images.append(image_array)
                    breeds.append(obj.name)
                del obj
            if data.data:
                del data.data
            del data
        del imagesData
        gc.collect()
        breeds = DataPreparator.__prepare_labels(breeds)
        return numpy.array(images), numpy.array(breeds)

    @staticmethod
    def __prepare_labels(labels):
        encoder = OneHotEncoder()
        encoded_labels = encoder.encode(labels)
        return encoded_labels # TODO: zrobić zapis encodera
