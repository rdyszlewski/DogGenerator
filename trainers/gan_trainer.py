from trainers.trainer import Trainer


class GanTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def train(self, data, labels, input_shape):
        discriminator, gan, generator = self._prepare_models(input_shape)
        x_train, x_test, y_train, y_test = self._prepare_data(data, labels)

    def __train(self, generator, discriminator, gan, x_train, batch_size, epochs):
        
        for e in range(epochs):
            print("Epoch %d" % e)
