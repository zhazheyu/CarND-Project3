from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D, SpatialDropout2D
from keras.layers.pooling import MaxPooling2D

def model_nvidia(img_shape):
    """Model based on the following paper by Nvidia
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=img_shape, output_shape=img_shape))
    # model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))    
    return model


# def model_udacity(img_shape):
#     model = Sequential()
#     model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=img_shape))
#     # model.add(Cropping2D(cropping=((70, 25),(0, 0))))
#     model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
#     model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
#     model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
#     model.add(Convolution2D(64, 3, 3, activation="relu"))
#     model.add(Convolution2D(64, 3, 3, activation="relu"))
#     model.add(Flatten())
#     model.add(Dense(100))
#     model.add(Dense(50))
#     model.add(Dense(10))
#     model.add(Dense(1))
#     return model


from utils import read_csv
from utils import generator
from utils import visualize_loss, visualize_preprocess, preprocess
from sklearn.model_selection import train_test_split

### Define file location
# CSV_PATH = '../DataSets/LastUsed/driving_log.csv' 
# IMG_PATH = '../DataSets/LastUsed/IMG/'
# TARGET_PATH = '../DataSets/LastUsed/PROCESSED/'

CSV_PATH = './data/driving_log.csv' 
IMG_PATH = './data/IMG/'
TARGET_PATH = './data/PROCESSED/'

### Parameter settings
batch_size=64
### These parameters need to be changed in drive.py
processed_image_shape = (75, 160)       # (75, 160)
input_shape = (75, 160, 3) # processed_image_shape + (3)  # 
colorspace='none'
### END: parameter settings

samples = read_csv(CSV_PATH)

### Only run 1 time.
### Preprocess images, such as change colorspace, crop, blur, and resize
### Processed image will be stored in TARGET_PATH
preprocess(samples, image_path=IMG_PATH, target_path=TARGET_PATH, output_shape=processed_image_shape, colorspace=colorspace)

### Take one image to visualize what preprocess is conducted
visualize_preprocess(samples, image_path=IMG_PATH)


### Model training
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('train_samples', len(train_samples))
print('validation_samples', len(validation_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, image_path=TARGET_PATH, batch_size=batch_size)
validation_generator = generator(validation_samples, image_path=TARGET_PATH, batch_size=batch_size)

model = model_nvidia(input_shape)
# model = model_udacity(input_shape)
# model.compile(optimizer=Adam(lr=0.001), loss='mse')
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)
model.save('temp.h5')

visualize_loss(history_object)