import os
import xml.etree.ElementTree as ET

from models import DogImage, ImageObject
from PIL import Image

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

        imagesData = []
        breeds_directories = os.listdir(annotation_path)
        for directory_name in breeds_directories:
            path = annotation_path + '/' + directory_name
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
