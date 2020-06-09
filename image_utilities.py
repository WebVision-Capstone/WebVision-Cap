""" image processing utilities """

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


train_dir = Path('d:\\data\\train')
val_dir = Path('d:\\data\\val')

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = .2,
    height_shift_range = .2,
    shear_range = .2,
    zoom_range = .2,
    horizontal_flip = True,
    fill_mode = 'nearest')

# don't augment the validation data!
test_datagen = ImageDataGenerator(
    rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150), # setting this to be small for now
    batch_size = 39, # try 39 because val_dir has 294099 images 
    class_mode = 'categorical')


val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size = (150, 150),
    batch_size = 39,
    class_mode = 'categorical')    

def limited_batch(generator, num):
    """ semi-functional : model will train as is, but better version in the works"""
    """ limits the number of batches the generator sends to the model """
    x = 0
    # example print-out that can be removed when used
    for data_batch, labels_batch in train_generator:
        
        # print('data batch shape: ', data_batch.shape)
        # print('labels batch shape: ', labels_batch.shape)
    #for i in train_generator:
        x += 1
        if x > num:
            break
        return data_batch, labels_batch


""" an implementation of LeNet as a toy model. It's small and trains faster. """
""" Not a final model """
class LeNet5(Model):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(6, 
                            kernel_size = (5,5), 
                            padding = 'same',
                            activation = 'relu')
        self.conv2 = Conv2D(16, 
                            kernel_size = (5,5),
                            padding = 'same',
                            activation = 'relu')
        self.max_pool = MaxPooling2D(pool_size = (2,2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation= 'relu')
        self.dense2 = Dense(84, activation = 'relu')
        self.dense3 = Dense(num_classes, activation  = 'softmax')
    def call(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        return x

toy_model = LeNet5(5000)
toy_model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#what is the input shape?
# (batch_size, height, width, rgb channels)
input_shape = (39, 150, 150, 3)
toy_model.build(input_shape)
toy_model.summary()
toy_model.fit(limited_batch(train_generator, 1000), 
              validation_data = limited_batch(val_generator, 1000),
              epochs = 10
              )