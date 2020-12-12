import numpy as np


class NoiseGenerator:

    @staticmethod
    def __add_noise_to_image(data, input_shape):
        # TODO: sprawdzić, czy to będzie działać
        for image in data:
            # noise = np.random.randint(5, size = TrainerConfig.input_shape, dtype = 'uint8')
            noise = np.random.normal(0, 0.1, size=input_shape)
            for i in range(input_shape[0]):
                for j in range(input_shape[1]):
                    for k in range(input_shape[2]):
                        if image[i][j][k] != 255:
                            image[i][j][k] += noise[i][j][k]
        return data
