# LipNet: End-to-End Sentence-level Lipreading
Keras implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas [Link](https://arxiv.org/abs/1611.01599).


# Data card
The model is trained on a subset of GRID Corpus.
The paper uses the entire GRID corpus to train on the dataset
Credits to [Nicholas Rennote](https://www.youtube.com/@NicholasRenotte)
The data link can be found in his [video](https://youtu.be/uKyojQjbx4c)

# Custom data
The project uses dlib to predict the lip landmarks so custom dataset are used
[DLib reference](http://dlib.net/)


## The project uses several dependencies

`pip install -r requirements.txt`
