import os
import subprocess
import config
from comet_ml import Experiment, ExistingExperiment
import etl
from alpha_gan import AlphaGAN

lr_gen = 0.02
lr_dis = 0.00001
latent_dim = 15
nb_epochs = 100000
samples_interval = nb_epochs//1000
batch_size=32
loss="binary_crossentropy"
path_to_data="data/creditcardfraud.zip"
test_size=0.01
random_state=42

hyper_params = {
    "lr_gen": lr_gen,
    "lr_dis": lr_dis,
    "latent_dim": latent_dim,
    "nb_epochs": nb_epochs,
    "samples_interval":samples_interval,
}

experiment = Experiment(project_name=config.project_name,api_key=config.api_key) # add your  api_key and name
experiment.log_parameters(hyper_params)

if __name__ == "__main__": 
        os.system("nvidia-smi -L")
        X_train, test_X, test_y  =etl.process_data(path_to_data,test_size,random_state)
        model = AlphaGAN(X_train=X_train,test_X=test_X,test_y=test_y,latent_dim=latent_dim,lr_dis=lr_dis,
        lr_gen=lr_gen,loss=loss,experiment=experiment)
        model.train_model(epochs=nb_epochs, batch_size=batch_size, samples_interval=samples_interval)

