# Dog Generator

The repository contains a neural network that creates images of dogs.
We use a generative adversarial network (GAN). A generative adversarial network is made up of two neural networks:
the generator, which learns to produce realistic fake data from a random seed. The fake examples produced by the generator are used as negative examples for training the discriminator
the discriminator, which learns to distinguish the fake data from realistic data.

Link to description and data: https://www.kaggle.com/c/generative-dog-images

## How use

Generating data
```
python cli.py [input_path] [output_path] [output_image_size] [batch_size]
```

Start
```
python cli.py train configuration_path
```