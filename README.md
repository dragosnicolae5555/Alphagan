# Alphagan

This an adapted implementation of AlphaGan  for Zero Shot outlier detection in https://www.kaggle.com/mlg-ulb/creditcardfraud
dataset using only negative samples for training 

###  Create conda environment with all dependencies 
``` conda env create -f environment.yaml ```

###  Activate conda environment 
```source activate alpha```

Add your comet-ml api key and project in config.py 

###  run main.py 
```python main.py```

Example experiment https://www.comet.ml/dragosnicolae5555/c-alpha/5f6db96a5eb248eba910c21148645334

General Framework :
 1.   Train GAN to generate only normal data points (negative samples).
 2.   When predicting anomaly, use GAN to reconstruct the input images of both normal and abnormal images (negative and positive samples).
 3.   Compute reconstruction  and discrimination losses.
    
    Discriminate between normal and abnormal cases using these statistics: 
    Reconstruction loss are the differences between original and reconstructed images.
    Discrimination loss is simply the output of the Discriminator.
    
    
    
    
