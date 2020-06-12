from data import DataPreparator
from image import ImageUtils
from loader import ImageLoader

data_path = "/media/roman/07765B7E452A5B73/Machine Learning/Dogs"

def main():
    data, labels = DataPreparator.prepare_data(data_path)
    print(data)


main()