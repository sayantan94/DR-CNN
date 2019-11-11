'''
Created on Nov 10, 2019

@author: sayantan

'''
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

from keras.models import Model

class Engine(object):
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    INPUT_SIZE =  (224, 224, 3)
    # FILES
    IMG_DEST = '/media/sayantan/HP v237w/sample/'
    TEST_CSV = '../Files/test.csv'
    TRAIN_CSV = '../Files/train.csv'
    
    
    def __init__(self):
        # THis is when the image directory creator file would be called
        self.trainCSV = pd.read_csv(self.TRAIN_CSV,dtype=str)
        self.testCSV = pd.read_csv(self.TEST_CSV,dtype=str)
        
        self.trainCSV['image'] = self.trainCSV['image'] + '.jpeg'
        self.testCSV['image'] = self.testCSV['image'] + '.jpeg'
         
    
    def model_definition(self):
        EPOCHS = 15
        BATCH_SIZE = 32
        STEPS_PER_EPOCH = 130
        VALIDATION_STEPS = 22
        POOL_SIZE = (2, 2)
        KERNEL_SIZE = (3, 3)
        INPUT_SHAPE = (224, 224, 3)
        TARGET_SIZE = (224, 224)
        """
        Class to outline the cnn model definition
        """

    
        
        model = Sequential()
#         
#         # First conv layer
#         model.add(Conv2D(64, KERNEL_SIZE, input_shape=INPUT_SHAPE,
#                          activation='relu', strides=1))
#         
#         # Second conv layer
#         model.add(Conv2D(64, KERNEL_SIZE, activation='relu',
#                          strides=1))
#         
#         # First pool layer
#         model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))
#         
#         # Third conv layer
#         model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
#                          strides=1))
#         
#         # Fourth conv layer
#         #model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
#         #                  strides=1, use_bias=True, padding='same'))
#         
#         # Second pool layer
#         model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))
#         
#         # Flattening
#         model.add(Flatten())
#         
#         # FC layers
#         #model.add(Dense(units=512, activation='relu'))
#         #model.add(Dense(units=5, activation='sigmoid'))
#         model.add(Dense(units=4, activation='sigmoid'))


        # First conv layer
        model.add(Conv2D(128, self.KERNEL_SIZE, input_shape=self.INPUT_SIZE,activation='relu',strides=1, padding='same'))
        #model.add(Conv2D(64, self.KERNEL_SIZE, input_shape=self.INPUT_SIZE,activation='relu',strides=1, padding='same'))
        #model.add(Conv2D(128, self.KERNEL_SIZE ,activation='relu',strides=1, padding='same'))
        
        #model.add(MaxPooling2D(pool_size=self.POOL_SIZE, strides=2))
        # 1st Pooling
        model.add(MaxPooling2D(pool_size=self.POOL_SIZE, strides=2))
        
        # Second Layer
        #model.add(Conv2D(128, self.KERNEL_SIZE, input_shape=self.INPUT_SIZE,activation='relu',strides=1, padding='same'))
        model.add(Conv2D(128, self.KERNEL_SIZE, activation='relu',strides=1, padding='same'))
        #model.add(Conv2D(128, self.KERNEL_SIZE, activation='relu',strides=1, padding='same'))
       
        # 2nd Pooling
        model.add(MaxPooling2D(pool_size=self.POOL_SIZE, strides=2))
        # Dropout
        #model.add(Dropout(0.1))
        
        # 3rd Layer
        #model.add(Conv2D(64, self.KERNEL_SIZE, input_shape=self.INPUT_SIZE,activation='relu',strides=1, padding='same'))
        #model.add(Conv2D(64, self.KERNEL_SIZE, input_shape=self.INPUT_SIZE,activation='relu',strides=1, padding='same'))
        # 3rd Pooling
        #model.add(MaxPooling2D(pool_size=self.POOL_SIZE, strides=2))
        # Dropout
        #model.add(Dropout(0.2))
        
        # FLatenning
        model.add(Flatten())
        
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        #print (dir(model))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=5, activation='sigmoid'))
        #model.add(Dense(5, activation='sigmoid'))
        #model.add(Dense(4, activation='sigmoid'))
        
        # Fully Connected Layer
        #model.add(Dense(units=512, activation='relu'))
        #model.add(Dense(units=4, activation='sigmoid'))
        #model.add(Dropout(0.2))
        #model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(5, activation='sigmoid'))
    
        return model
        
    def compile_model(self,model):
        """
        Method to compile the model definition and the history of model.
        
        loss='categorical_crossentropy'
        """
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        print(model.summary())
        
        return True
    
    
    def train_model(self, model):
        # Load the data from the data-frames
        print ('Initializing Data Sets')
        train_datagen=ImageDataGenerator(rescale=1./255.) # Training Images
        test_datagen=ImageDataGenerator(rescale=1./255.)  # Testing Images
        
        # Training Set Definition
        self.trainCSV['level'] = self.trainCSV['level'].astype(str)
        self.testCSV['level'] = self.testCSV['level'].astype(str)
        
        print (type(self.trainCSV['level']))
        training_set = train_datagen.flow_from_dataframe(
                        dataframe=self.trainCSV,
                        directory = self.IMG_DEST, 
                        x_col="image", 
                        y_col="level", 
                        batch_size=2,
                        class_mode='categorical',
                        shuffle = True, 
                        target_size=(224,224)
                        )
        
        validation_set = test_datagen.flow_from_dataframe(
                dataframe=self.testCSV,
                directory = self.IMG_DEST, 
                x_col="image", 
                y_col="level", 
                class_mode='categorical',
                batch_size=2,
                shuffle = True, 
                target_size=(224,224)
                )
        
        # Training the model and generating desired out-put
        print(11)
        history = model.fit_generator(training_set,
                                   steps_per_epoch =15,
                                   epochs = 20,
                                   validation_data = validation_set,
                                   validation_steps = 50)
        
        
        print(22)
        return history
    
     
    def show_layers(self,model):
        # Take any random image and try to predict the class
        pass
    
    def predict_image(self,model):
        #img_path = 'test_set/triangles/drawing(2).png'
        
        img = image.load_img('%s'%self.IMG_DEST+'1032_right'+'.jpeg',  target_size = (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        plt.imshow(img_tensor[0])
        plt.show()
    
        print(img_tensor.shape) 
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        print("Predicted class is:",classes)
        
        
        layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
        activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
        # Load an image and pass the image
        activations = activation_model.predict(img_tensor)
        first_layer_activation = activations[0]
        print(first_layer_activation.shape)
        plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
        
        plt.show()
        
        
        layer_names = []
        for layer in model.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        images_per_row = 16
        
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
        
    def model_metrics(self, history):
        """
        Method to generate the class performance and return the results
        """
        print (dir(history))
        print (history.history)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        
        plt.legend()
        plt.show()
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.figure()
        
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.show()
            
        
        

if __name__ == '__main__':
    obj = Engine()
    
    # Define Mode
    model = obj.model_definition()
    # Model Summary
    obj.compile_model(model)
    # Train Model
    history = obj.train_model(model)
    # Plot Graph
    #print (history)
    obj.model_metrics(history)
    
    obj.predict_image(model)
    
    
    
        
    