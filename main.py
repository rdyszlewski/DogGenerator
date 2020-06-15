from data import DataPreparator
from trainers.simple_batch import SimpleBatchTrainer

data_path = "/media/roman/07765B7E452A5B73/Machine Learning/Dogs"

def main():
    input_shape = (128, 128, 3)
    data, labels = DataPreparator.prepare_data(data_path, input_shape[0], input_shape[1])
    trainer = SimpleBatchTrainer()
    trainer.train(data, labels, input_shape)




main()