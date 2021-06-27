import os

# specify the shape of the inputs for our network
IMG_SHAPE = (224, 224, 3)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100

# define the path to the base output directory
BASE_OUTPUT = "Output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])