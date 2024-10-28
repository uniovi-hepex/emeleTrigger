import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences




def get_training_dataset(hdf5_path, BATCH_SIZE=128):

    # Get the point clouds
    x_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/point_clouds')
    # Get the original points
    y_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/images')
    # Zip them to create pairs
    training_dataset = tf.data.Dataset.zip((x_train,y_train))
    # Apply the data transformations
    #training_dataset = training_dataset.map(resize_and_format_data)
    
    # Shuffle, prepare batches, etc ...
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)
    
    # Return dataset
    return training_dataset

