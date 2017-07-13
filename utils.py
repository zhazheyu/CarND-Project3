import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_csv(csv_path):
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        # train_samples, validation_samples = train_test_split(lines, test_size=0.2)
        return lines


def preprocess(samples, image_path='./data/IMG/', target_path='./data/PROCESSED/', output_shape=(75, 160), colorspace='ss'):
    num_samples = len(samples)
    for i in range(num_samples):
        sample = samples[i]
        name = sample[0].split('/')[-1]
        center_image = cv2.imread(image_path+name)
        processed = preprocess_image(center_image, output_shape=output_shape, colorspace=colorspace)[0]

        # plt.imshow(processed)
        # plt.show()      
        cv2.imwrite(target_path + name, processed)
        if i % 500 == 0:
            print('finished %s of %s', i, num_samples)


def generator(samples, image_path='./data/IMG/', batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffled_arrays = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_arrays[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = image_path+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Augment data with mirror images, so batch_size x 2
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print('X_train size: ', X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


### image processing
def crop_img(img):
    """Returns croppped image
    """
    return img[65:140, ]

def blur_img(image, kernel_size = 3):
    """ kernel_size Must be an odd number (3, 5, 7...)
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    

# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.
    
#     Only keeps the region of the image defined by the polygon
#     formed from `vertices`. The rest of the image is set to black.
#     """
#     #defining a blank mask to start with
#     mask = np.zeros_like(img)   
    
#     #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
        
#     #filling pixels inside the polygon defined by "vertices" with the fill color    
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
    
#     #returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image
#     imshape = image.shape
#     vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)    


def preprocess_image(img, output_shape=(75, 160), colorspace='none'):
    # return img, colorspace
    colorspace = 'none'
    """
    Reminder:

    Source image shape is (160, 320, 3)
    My preprocessing algorithm consists of the following steps:
    1. Converts BGR to YUV colorspace.
        This allows us to leverage luminance (Y channel - brightness - black and white representation),
        and chrominance (U and V - blue–luminance and red–luminance differences respectively)
    2. Crops top 31.25% portion and bottom 12.5% portion.
        The entire width of the image is preserved.
        This allows the model to generalize better to unseen roadways since we crop
        artifacts such as trees, buildings, etc. above the horizon. We also clip the
        hood from the image.
    3. Finally, I allow users of this algorithm the ability to specify the shape of the final image via
        the output_shape argument.
        Once I've cropped the image, I resize it to the specified shape using the INTER_AREA
        interpolation algorithm as it is the best choice to preserve original image features.
        See `Scaling` section in OpenCV documentation:
        http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """
    # convert image to another colorspace
    if colorspace == 'yuv':
        image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif colorspace == 'hsv':
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'hls':
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif colorspace == 'rgb':
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        image = img

    
    # The entire width of the image is preserved
    cropped = crop_img(image)
    # Let's blur the image to smooth out some of the artifacts
    blurred = blur_img(cropped, kernel_size=3)
    # blurred = cropped
    # resize image to output_shape
    img = cv2.resize(blurred, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)
    return img, cropped, blurred, image


def visualize_preprocess(samples, image_path='./data/IMG/', colorspace='none'):
    sample = samples[0]
    name = image_path+sample[0].split('/')[-1]
    # name = image_path+'center_2017_07_08_14_01_57_330.jpg'
    # name = './san_francisco.jpg'
    center_image = mpimg.imread(name)
    plt.subplot(1, 5, 1)
    plt.imshow(center_image)
    plt.title("original")
    # plt.axis('off')
    
    img, cropped, blurred, colorspace = preprocess_image(center_image, colorspace=colorspace)

    plt.subplot(1, 5, 2)
    plt.imshow(colorspace)
    plt.title("colorspace")    
    plt.subplot(1, 5, 3)
    plt.imshow(cropped)
    plt.title("cropped")
    plt.subplot(1, 5, 4)
    plt.imshow(blurred)
    plt.title("blurred")
    plt.subplot(1, 5, 5)
    plt.imshow(img)
    plt.title("resized")
    plt.show()


### Post-processing

def visualize_loss(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
