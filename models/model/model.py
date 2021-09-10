"""New Project Example

This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any
from layer import Featureset, Train, Dataset
from PIL import Image
import io
import base64
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping

def train_model(train: Train, ds:Dataset("catsdogs"), pf: Featureset("cat_and_dog_features")) -> Any:
    # train: Train, df:Dataset("cats-and-dogs-classification"), pf: Featureset("animal_features")
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml

    Returns:
       model: Trained model object

    """
    df = ds.to_pandas().merge(pf.to_pandas(), on='id')
    training_set = df[(df['path'] == 'training_set/dogs') | (df['path'] == 'training_set/cats')]
    testing_set = df[(df['path'] == 'test_set/dogs') | (df['path'] == 'test_set/cats')]
    X_train = np.stack(training_set['content'].map(load_process_images))
    X_test = np.stack(testing_set['content'].map(load_process_images))

    train.register_input(X_train)
    train.register_output(df['category'])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1
                                       )
    train_datagen.fit(X_train)
    training_data = train_datagen.flow(X_train, training_set['category'], batch_size=32)
    validation_gen = ImageDataGenerator(rescale=1. / 255)
    testing_data = validation_gen.flow(X_test, testing_set['category'], batch_size=32)

    # data_augmentation = keras.Sequential(
    #     [keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    #      keras.layers.experimental.preprocessing.RandomRotation(0.1),
    #      ]
    # )

    base_model = Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False)

    base_model.trainable = False
    inputs = keras.Input(shape=(150, 150, 3))
    # x = data_augmentation(inputs)
    x = keras.applications.xception.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=keras.metrics.BinaryAccuracy())
    model.fit(training_data, epochs=20, validation_data=testing_data)

    base_model.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=keras.metrics.BinaryAccuracy())

    callbacks = [
        EarlyStopping(patience=5),
    ]

    model.fit(training_data, epochs=15, validation_data=testing_data, callbacks=callbacks)

    test_loss, test_accuracy = model.evaluate(testing_data)
    train_loss, train_accuracy = model.evaluate(training_data)

    train.log_metric("Testing Accuracy", test_accuracy)
    train.log_metric("Testing Loss", test_loss)

    train.log_metric("Training Accuracy", train_accuracy)
    train.log_metric("Training Loss", train_loss)
    return model


def load_process_images(content):
    image_decoded = base64.b64decode(content)
    image = Image.open(io.BytesIO(image_decoded)).resize([150, 150])
    image = img_to_array(image)
    return image
