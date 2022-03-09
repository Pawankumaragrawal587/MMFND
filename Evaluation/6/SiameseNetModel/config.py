import os

# specify the shape of the inputs for our network
IMG_SHAPE = (1, 4096)

TXT_SHAPE = (1024,1)

# specify the batch size and number of epochs
BATCH_SIZE = 32
EPOCHS = 30

# define the path to the base output directory
BASE_OUTPUT = "Output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
