## Parameters
NB_EPOCHS = 30
IMAGE_SIZE	= 64
SCAN_CROSS_SECTION_SIZE = 512
NUM_SLICES = 10
BATCH_SIZE = 128
DROPOUT_PROB = 0.25

NUM_SAMPLES_PER_POSITIVE_SAMPLE_PER_SAMPLE = 400  # This variable controls how many positive samples are generated for every single positive slie in the scan

PATH_TO_GENERATE_TRAINING_DATA = 'TRAIN_DATA3'  # The new training dataset will be generated here

PATH_TO_SAVE_INTERMEDIATE_DATA = 'mid_data'

THRESHOLD_VALUE = -300
