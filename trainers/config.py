class TrainerConfig:
    # depth = 32
    # epochs = 1000
    # batch_size = 64

    depth = 128
    epochs = 1000
    batch_size = 16

    input_shape = (64, 64, 3)
    noise_size = 100

    data_path = '/mnt/07765B7E452A5B73/Machine Learning/Dogs/processing'
    output_path = 'output/gan_generated_image %d.png'
