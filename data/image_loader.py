import os
import xml.etree.ElementTree as ET

from data.models import DogImage, ImageObject
from PIL import Image
from multiprocessing.pool import Pool


class ImageLoader:

    def load(self, directory_path):
        return self.__load_images_data(directory_path)


    def __load_breeds(self, directory_path):
        path = directory_path + "/Annotation"
        breeds_text = os.listdir(path)
        result = {}
        for text in breeds_text:
            code, name = text.split('-', 1)
            name = name.replace('_', ' ')
            result[code] = name
        return result

    def __load_images_data(self, directory_path):
        annotation_path = directory_path + '/Annotation'
        images_path = directory_path + '/all-dogs'
        breeds_directories = os.listdir(annotation_path)

        parameters = self.__prepare_parameters(annotation_path, breeds_directories, images_path)
        result = self.__start_loading_data(parameters)
        return result



    def __start_loading_data(self, parameters):
        # TODO: ustawić jakieś mądre ustawianie wątków. Duże prawdopodobieństwo, że braknie pamięci
        # threads = ThreadsUtils.get_available_threads()
        threads = 12
        with Pool(processes=threads) as pool:
            loading_results = pool.map(self._load_images, parameters)
        concatenated_result = sum(loading_results, [])
        for element in loading_results:
            del element
        return concatenated_result

    def __prepare_parameters(self, annotation_path, breeds_directories, images_path):
        parameters = []
        for directory_name in breeds_directories:
            path = annotation_path + '/' + directory_name
            parameters.append((path, images_path))
        return parameters

    def _load_images(self, parameters):
        path = parameters[0]
        images_path = parameters[1]
        imagesData = []
        for filename in os.listdir(path):
            filepath = path + "/" + filename
            dogImage = self.__load_annotation_file(filepath, images_path)
            imagesData.append(dogImage)
        return imagesData


    def __load_image(self, path):
        if os.path.isfile(path):
            image = Image.open(path)
            return image

    def __load_annotation_file(self, path, images_path):
        root = ET.parse(path).getroot()

        dogImage = DogImage()
        dogImage.name = root.find('filename').text

        size = root.find('size')
        dogImage.width = int(size.find('width').text)
        dogImage.height = int(size.find('height').text)

        objects = root.findall('object')
        for obj in objects:
            imageObject = self.__get_object(obj)
            dogImage.objects.append(imageObject)
        imagePath = images_path + '/' + dogImage.name + '.jpg'
        dogImage.data = self.__load_image(imagePath)
        return dogImage


    def __get_object(self, obj):
        imageObject = ImageObject()
        imageObject.name = obj.find('name').text
        imageObject.pose = obj.find('pose').text
        imageObject.truncated = True if obj.find('truncated').text == '1' else False
        box = obj.find('bndbox')
        imageObject.minX = int(box.find('xmin').text)
        imageObject.minY = int(box.find('ymin').text)
        imageObject.maxX = int(box.find('xmax').text)
        imageObject.maxY = int(box.find('ymax').text)

        return imageObject
