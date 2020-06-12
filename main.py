from loader import ImageLoader

data_path = "/media/roman/07765B7E452A5B73/Machine Learning/Dogs"

def main():
    loader = ImageLoader()
    images = loader.load(data_path)
    print(images)

main()