import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from test_model import test_model 

class AlphaGAN():
    def __init__(self,latent_dim,lr_dis,lr_gen,loss,experiment,test_X,test_y,X_train):
        self.test_X=test_X
        self.test_y=test_y
        self.X_train=X_train
        self.x_shape = (self.X_train.shape[1],)
        self.latent_dim = latent_dim
        self.loss=loss
        self.experiment=experiment
        __optimizer = Adam(lr_dis, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.loss],optimizer=__optimizer,metrics=['accuracy'])
        optimizer = Adam(lr_gen, 0.5)

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        x_ = self.generator(z)

        # Encode image
        x = Input(shape=self.x_shape)
        z_ = self.encoder(x)
        reconstructed_x = self.generator(z_)

        # Latent -> x is fake, and x -> latent is valid
        fake = self.discriminator([z, x_])
        valid = self.discriminator([z_, x])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.alphagan_generator = Model([z, x], [fake, valid,reconstructed_x])
        self.alphagan_generator.compile(loss=[self.loss, self.loss,self.loss],
            optimizer=optimizer)

    def train_model(self, epochs, batch_size, samples_interval):
        with self.experiment.train():

            # Adversarial ground truths
            valid = np.ones((batch_size, 1))*0.9
            fake = np.zeros((batch_size, 1))

            for epoch in range(epochs):


                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise and generate x
                z = np.random.normal(size=(batch_size, self.latent_dim))
                x_ = self.generator.predict(z)

                # Select a random batch of images and encode
                idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                x = self.X_train[idx]
                z_ = self.encoder.predict(x)
                reconstructed_x = self.generator.predict(z_)

                # Train the discriminator (x -> z is valid, z -> x is fake)
                d_loss_real = self.discriminator.train_on_batch([z_, x], valid)
                d_loss_fake = self.discriminator.train_on_batch([z, x_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (z -> x is valid and x -> z is is invalid)
                g_loss = self.alphagan_generator.train_on_batch([z, x], [valid, fake,x])

                # Plot the progress
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
                self.experiment.log_metric("Discriminator Loss", d_loss[0], step=epoch)
                self.experiment.log_metric("Generator+Reconstruction Loss", g_loss[0], step=epoch)
                self.experiment.log_metric("Discrimination Accuracy", 100*d_loss[1], step=epoch)

                # If at save interval => save generated x samples
                if (epoch+1) % samples_interval == 0:
                    self.sample_interval(epoch,self.test_X,self.test_y, self.experiment)

    def sample_interval(self,epoch,test_X,test_y,experiment):
        test_model(self,epoch,test_X,test_y, self.experiment)


    def build_encoder(self):
        model = Sequential()

        model.add(Dense(64,input_shape=self.x_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))
        model.name="encoder"      
        model.summary()

        x = Input(shape=self.x_shape)
        z = model(x)

        return Model(x, z)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.x_shape), activation='tanh'))
        model.add(Reshape(self.x_shape))
        model.name="generator"  
        model.summary()
        z = Input(shape=(self.latent_dim,))
        gen_x = model(z)
        return Model(z, gen_x)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim, ))
        x = Input(shape=self.x_shape)
        d_in = concatenate([z, x])

        model = Dense(128)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)
        model = Model([z, x], validity)
        model.name="discriminator"
        model.summary()
        return model

    def get_losses(self,x):
        with self.experiment.test():
            batch_size = x.shape[0]
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            z = np.random.normal(size=(batch_size, self.latent_dim))
            x_ = self.generator.predict(z)

            # Select and encode

            z_ = self.encoder.predict(x)
            reconstructed_x = self.generator.predict(z_)

            # Train the discriminator (x -> z is valid, z -> x is fake)
            d_loss_real = self.discriminator.evaluate([z_, x], valid,verbose=0)
            d_loss_fake = self.discriminator.evaluate([z, x_], fake,verbose=0)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (z -> x is valid and x -> z is is invalid)
            g_loss = self.alphagan_generator.evaluate([z, x], [valid, fake,reconstructed_x],verbose=0)
            return d_loss[0] + g_loss[0]