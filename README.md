# ali-rsi-pendulum

TODO:

1) Install miniconda on your computer with Python 3.8 from this link: https://docs.conda.io/en/latest/miniconda.html.
2) Create a new conda environment and install the required packages: 
   ```
   conda init
   conda create -n pendulum
   conda activate pendulum
   pip install torch torchvision torchaudio scipy matplotlib
   ``` 
   It might be possible that additional packages would be needed later. You can install them with `pip` analogously.
3) Start reading and playing with `pendulum.py`. 
   We will take a top-down approach, which means that you will have lots of questions about the code.
   The tutorial below might be a good starting point. There are more pytorch tutorials out there.
   https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
   You should keep a list with questions and ask Peter and I about them. 
   A rule of thumb is that if you cannot solve the problem for 30 minutes, ask us.
   
4) Your first task is to understand the `PendulumDataset` and convert the dataset into 2D images.
   Right now the dataset representation for the orbits is `[angle, angular_momentum]`.
   The neural network is a simple multi layer perceptron (MLP).
   We want to make the dataset to be 2D images, and the neural network to be a convolutional neural network (CNN).

5) In general, you should keep another list with your ideas about the research. We can discuss the ideas together.
