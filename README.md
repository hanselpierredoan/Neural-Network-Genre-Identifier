# Neural-Network-Genre-Identifier
Train a neural network to identify the music genre that is inputted. 

It defines a neural network that can be used to predict the genre of a song based on its features. The network is trained on a dataset of songs, and then evaluated on a test set of songs. The accuracy of the network is printed out. The network is then saved so that it can be used to predict the genre of new songs.

It uses the TensorFlow network sample library and pulls the data from GITZAN. 

I am loading a dataset, which is a list of string arrays. I am then creating a neural network. The neural network has an input layer, which is the song features, and an output layer, which is the genre prediction. Then, I'm splitting the dataset into a train set and a test set. The train set is 80% of the dataset and the test set is 20%. We are then converting the train set and the test set into float arrays. The float arrays are the train features and the test features. Then we are creating the neural network. The network has a hidden layer and an output layer. The hidden layer has 10 nodes and the output layer has 10 nodes. We are then training the network. The network is trained using the gradient descent algorithm. The network is trained for 10000 iterations. We are then evaluating the model. The model is evaluated on the test set. The accuracy of the model is 0.8.

The dataset is loaded from a file called "GITZAN.txt". The neural network has 10 input nodes, 10 hidden nodes, and 10 output nodes. The network is trained using gradient descent with a learning rate of 0.5. The model is then evaluated on a test set. The accuracy of the model is printed to the console. Finally, the model is saved to a file called "model/".

This code splits it into a train and test set, and creates a neural network to predict the genre of a song. The code above loads in a dataset from a file called GITZAN.txt. This dataset contains information on songs, including their length, tempo, and genre. The code then splits the dataset into a training set and a test set. The training set is used to train the neural network, and the test set is used to evaluate the accuracy of the neural network.

The neural network has an input layer, a hidden layer, and an output layer. The input layer is the song features, the hidden layer is the prediction of the genre, and the output layer is the actual genre. The code trains the neural network on the train set and then evaluates the accuracy on the test set. Finally, the code saves the model.

The hidden layer is used to learn the relationships between the song features and the genres. The input layer contains the song features, and the output layer predicts the genre of the song. The output layer of the code above is a softmax layer that outputs the probabilities for each class which contains the output weights and biases. The code trains the network by iteratively passing batches of song features and labels to the network. The network then adjusts the weights and biases of the hidden and output layers so that it can better predict the genres of the songs.
