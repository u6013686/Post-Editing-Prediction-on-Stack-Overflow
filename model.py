from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_util import load_data
import numpy as np

def train_model(pattern, batch_size, epochs):
    """
    This function will generate a CNN model and fit the data of edit type (pattern),
    then save the model for visualization and save training and test data for predicting
    testing data other training methods.
    :param pattern: edit type
    :return: None
    """

    # hyperparameters
    embedding_dim = 128
    filter_sizes = [3, 4, 5]
    num_filters = 512
    drop = 0.5

    #load data
    print('Loading data')
    x, y, _, vocabulary_inv = load_data(pattern)
    sequence_length = x.shape[1]
    vocabulary_size = len(vocabulary_inv)

    #split data into training data, testing data and validation data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # save training and test data for predicting testing data and other training methods
    np.savetxt('splitted_data/' + pattern + 'x_train.txt', x_train)
    np.savetxt('splitted_data/' + pattern + 'y_train.txt', y_train)
    np.savetxt('splitted_data/' + pattern + 'x_test.txt', x_test)
    np.savetxt('splitted_data/' + pattern + 'y_test.txt', y_test)

    # build CNN model layer by layer
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # create the model
    model = Model(inputs=inputs, outputs=output)

    #checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # optimizer

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # train the model by fitting the data into the model
    print("Training Model")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation))

    # display the structure of model
    model.summary()

    #save the model
    print("Save model to disk")
    model_json = model.to_json()
    with open(pattern +"model.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights(pattern + "model.hdf5")

