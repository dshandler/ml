import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import cv2
import os
from imutils import paths
import imutils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')

class SimplePreprocessor:
    def __init__(self, width, height, inter= cv2.INTER_AREA):
        self.width= width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        try:
            image= cv2.resize(image, (self.width, self.height), interpolation = self.inter)
            b,g,r= cv2.split(image)
            image= cv2.merge((r,g,b))
            return image
        except Exception:
            pass
    
        
        
    
class AnimalsDatasetManager:
    def __init__(self, preprocessors=None, random_state=6789):
        self.random = np.random.RandomState(random_state)
        self.preprocessors = preprocessors
        # self.preprocessors is a list of preprocessor for data augmentation
        # it can be an instance of SimplePreprocessor, which performs resizing image and re-orders the channels to RGB
        if self.preprocessors is None:
            self.preprocessors = list()
    
    def load(self, label_folder_dict, max_num_images=500, verbose =-1):
        # label_folder_dict: a dict mapping label to folder path
        data =list(); labels = list()
        for label, folder in label_folder_dict.items():
            image_paths = list(paths.list_images(folder)) # get the list of paths to all images in the folder
            print(label, len(image_paths))
            j = 0
            for (i, image_path) in enumerate(image_paths):
                image = cv2.imread(image_path)
                #if preprocessing images
                if self.preprocessors is not None: 
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                data.append(image); labels.append(label)
                if verbose > 0 and i>0 and (i+1)% verbose ==0:
                    print("Processed {}/{}".format(i+1, max_num_images))
                #if i+1 >= max_num_images:
                #    break
        self.data= np.array(data)
        self.labels= np.array(labels)
        self.train_size= int(self.data.shape[0])
    
    def process_data_label(self):
        label_encoder= preprocessing.LabelEncoder()
        label_encoder.fit(self.labels)
        self.labels= label_encoder.transform(self.labels)
        self.data= self.data.astype("float") / 127.5 - 1 # standardize pixel value to range [-1, 1]
        self.classes= label_encoder.classes_
        
    
    def train_valid_test_split(self, model=None, train_size=0.8, test_size= 0.1, rand_seed=33):
        if model is not None:
            valid_size = 1 - (train_size + test_size)
            X1, X_test, y1, y_test = train_test_split(self.data, self.labels, test_size = test_size, random_state= rand_seed)
            X_test_adv = X_test
            X_test_adv1 = self.attack_model(model, 'fgsm', X_test, y_test)
            X_test_adv2 = self.attack_model(model, 'pgd', X_test, y_test)
            X_test_adv3 = self.attack_model(model, 'mim', X_test, y_test)
            X_test_adv = np.append(X_test_adv, X_test_adv1,axis=0)
            X_test_adv = np.append(X_test_adv, X_test_adv2,axis=0)
            X_test_adv = np.append(X_test_adv, X_test_adv3,axis=0)
            self.X_test= X_test_adv
            y_test_adv = y_test 
            y_test_adv = np.append(y_test_adv, y_test)
            y_test_adv = np.append(y_test_adv, y_test)
            y_test_adv = np.append(y_test_adv, y_test)
            self.y_test = y_test_adv
            print("Test Processed")
            X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
            X_batches = X_train
            batch_size = 500
            for idx_start in range(0, X_train.shape[0], batch_size):
                idx_end = min(X_train.shape[0], idx_start + batch_size)
                batch_image = X_train[idx_start:idx_end]
                batch_label = y_train[idx_start:idx_end]
                X_train_adv1 = self.attack_model(model, 'fgsm', batch_image, batch_label)
                X_train_adv2 = self.attack_model(model, 'pgd', batch_image, batch_label)
                X_train_adv3 = self.attack_model(model, 'mim', batch_image, batch_label)
                X_batches = np.append(X_batches, X_train_adv1,axis=0)
                X_batches = np.append(X_batches, X_train_adv2,axis=0)
                X_batches = np.append(X_batches, X_train_adv3,axis=0)
            self.X_train= X_batches
            y_train_adv = y_train
            y_train_adv = np.append(y_train_adv, y_train)
            y_train_adv = np.append(y_train_adv, y_train)
            y_train_adv = np.append(y_train_adv, y_train)
            self.y_train = y_train_adv
            print("Train Processed")
            X_valid_adv = X_valid
            X_valid_adv1 = self.attack_model(model, 'fgsm', X_valid, y_valid)
            X_valid_adv2 = self.attack_model(model, 'pgd', X_valid, y_valid)
            X_valid_adv3 = self.attack_model(model, 'mim', X_valid, y_valid)
            X_valid_adv = np.append(X_valid_adv, X_valid_adv1,axis=0)
            X_valid_adv = np.append(X_valid_adv, X_valid_adv2,axis=0)
            X_valid_adv = np.append(X_valid_adv, X_valid_adv3,axis=0)
            
            self.X_valid= X_valid_adv
            y_valid_adv = y_valid
            y_valid_adv = np.append(y_valid_adv, y_valid)
            y_valid_adv = np.append(y_valid_adv, y_valid)
            y_valid_adv = np.append(y_valid_adv, y_valid)
            self.y_valid = y_valid_adv
            print("Validation Processed")
        else:
            valid_size = 1 - (train_size + test_size)
            X1, X_test, y1, y_test = train_test_split(self.data, self.labels, test_size = test_size, random_state= rand_seed)
            self.X_test= X_test
            self.y_test= y_test
            X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
            self.X_train= X_train
            self.y_train= y_train
            self.X_valid= X_valid
            self.y_valid= y_valid
    
    def next_batch(self, batch_size=32):
        idx = self.random.choice(self.X_train.shape[0], batch_size, replace=batch_size > self.X_train.shape[0])
        return self.X_train[idx], self.y_train[idx]

    def attack_model(self, model, attack, input_image, input_label, epsilon = 0.0313, num_steps = 20, step_size = 0.005, clip_value_min=0, clip_value_max=1.0, gamma=0.9):
        
        input_image = tf.convert_to_tensor(input_image)
        input_label = tf.convert_to_tensor(input_label)
        loss_fn = tf.keras.losses.sparse_categorical_crossentropy
        if attack == 'fgsm':
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                pred = model(input_image)
                loss = loss_fn(input_label, pred)
            gradient = tape.gradient(loss, input_image)
            signed_grad = tf.sign(gradient)
            adv_image = input_image + epsilon * tf.sign(gradient)
            adv_image = tf.clip_by_value(adv_image, clip_value_min, clip_value_max)
            adv_image = tf.stop_gradient(adv_image)
        
        elif attack == 'pgd':
            random_noise = tf.random.uniform(
                shape=input_image.shape, 
                minval=-epsilon, 
                maxval=epsilon,
                dtype=tf.dtypes.float64
            )
            adv_image = input_image + random_noise
            for i in range(num_steps):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(adv_image)
                    pred = model(adv_image)
                    loss = loss_fn(input_label, pred)
                gradient = tape.gradient(loss, adv_image)
                adv_image = adv_image + step_size * tf.sign(gradient)
                adv_image = tf.clip_by_value(adv_image, input_image-epsilon, input_image+epsilon) 
                adv_image = tf.clip_by_value(adv_image, clip_value_min, clip_value_max)
                adv_image = tf.stop_gradient(adv_image)
        
        elif attack == 'mim':
            random_noise = tf.random.uniform(
                shape=input_image.shape, 
                minval=-epsilon, 
                maxval=epsilon,
                dtype=tf.dtypes.float64
            )
            adv_image = input_image + random_noise
            adv_noise = random_noise
            for i in range(num_steps):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(adv_image)
                    pred = model(adv_image)
                    loss = loss_fn(input_label, pred)
                gradient = tape.gradient(loss, adv_image)
                adv_image_new = adv_image + step_size * tf.sign(gradient) 
                adv_image_new = tf.clip_by_value(adv_image_new, input_image-epsilon, input_image+epsilon) 
                adv_image_new = tf.clip_by_value(adv_image_new, clip_value_min, clip_value_max) 
                adv_noise = gamma*adv_noise + (1-gamma)*(adv_image_new - adv_image)
                adv_image = adv_image_new
                adv_image = tf.stop_gradient(adv_image)
            adv_image = adv_image + adv_noise
            adv_image = tf.clip_by_value(adv_image, input_image-epsilon, input_image+epsilon) 
            adv_image = tf.clip_by_value(adv_image, clip_value_min, clip_value_max)
            
        return adv_image.numpy()


class DefaultModel():
    def __init__(self,
                 name='network1',
                 width=32, height=32, depth=3,
                 num_blocks=2,
                 feature_maps=32,
                 num_classes=4, 
                 drop_rate=0.2,
                 batch_norm = None,
                 is_augmentation = False,
                 activation_func='relu',
                 optimizer='adam',
                 batch_size=10,
                 num_epochs= 20,
                 learning_rate=0.0001,
                 verbose= True):
        assert (1 << num_blocks <= min(width, height))
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.num_blocks = num_blocks
        self.feature_maps = [feature_maps * (1 << i) for i in range(num_blocks)]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.is_augmentation = is_augmentation
        self.activation_func = activation_func
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        if optimizer == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate)
        elif optimizer == 'nadam':
            self.optimizer = keras.optimizers.Nadam(learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = keras.optimizers.Adagrad(learning_rate)
        elif optimizer== 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate)
        elif optimizer == 'adadelta':
            self.optimizer = keras.optimizers.Adadelta(learning_rate)
        else:
            self.optimizer = keras.optimizers.SGD(learning_rate)
        self.model = models.Sequential()
        self.history = None
    
    def build_cnn(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation=self.activation_func, input_shape=(32,32,3)))
        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation=self.activation_func))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation=self.activation_func))
        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation=self.activation_func))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def summary(self):
        print(self.model.summary())
    
    def fit(self, data_manager, batch_size=None, num_epochs=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x = data_manager.X_train, y = data_manager.y_train, validation_data = (data_manager.X_valid, data_manager.y_valid), 
                                      epochs = num_epochs, batch_size = batch_size, verbose= self.verbose)
    
    def compute_accuracy(self, X_test, y_test, batch_size = 64):
        _, acc= self.model.evaluate(X_test, y_test, batch_size = batch_size)
        return acc
    
    def plot_progress(self, ylim=[0.6, 2.5]):
        #pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        #plt.grid(True)
        #plt.gca().set_ylim(0, 3) # set the vertical range to [0-1.5]
        #plt.show()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.history.history['accuracy'], label='train accuracy', color='green', marker="o")
        ax1.plot(self.history.history['val_accuracy'], label='valid accuracy', color='blue', marker = "v")
        ax2.plot(self.history.history['loss'], label = 'train loss', color='orange', marker="o")
        ax2.plot(self.history.history['val_loss'], label = 'valid loss', color='red', marker = "v")
        ax1.legend(loc=3)

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')
        ax2.legend(loc=4)
        plt.ylim(ylim) #[0.6, 2.5]
        plt.show()

    def predict(self, X):
        probs=  self.model.predict(X)
        y_preds= np.argmax(probs, axis =1)
        return y_preds
    
    def plot_prediction(self, X, y, classes, tile_shape=(5, 5)):
        y_pred= self.predict(X)
        plt.clf()
        fig, ax = plt.subplots(tile_shape[0], tile_shape[1], figsize=(3 * tile_shape[1], 3 * tile_shape[0]))
        idx = np.random.choice(len(y_pred), tile_shape[0] * tile_shape[1])

        for i in range(tile_shape[0]):
            for j in range(tile_shape[1]):
                ax[i, j].imshow((X[idx[i * tile_shape[1] + j]] + 1.0)/2)
                ax[i, j].set_title('{} (p: {})'.format(classes[y[idx[i * tile_shape[1] + j]]],
                                                        classes[y_pred[idx[i * tile_shape[1] + j]]]))
                ax[i, j].grid(False)
                ax[i, j].axis('off')
        plt.show()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
        
                    
     
            
                   
        
