# pianoroll-vq-vae

this is inspire from https://github.com/zalandoresearch/pytorch-vq-vae

we modify the model to let it able to encode and decode pianorolls

we use the LPD-cleansed dataset from Lakh Pianoroll Dataset, which you can get from https://salu133445.github.io/lakh-pianoroll-dataset/dataset

unzip the dataset and create a ./data folder to put it in 

run the train.py to see the training result 

TODO:

modify the dataset to let it get object at getitem() but not at initialization

add tools to visualise the results
