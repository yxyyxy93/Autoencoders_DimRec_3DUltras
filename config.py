epochs = 100
num_channel = 256

image_dir = r'.\dataset\sim_data'  # path to the 'sim_data' directory
label_dir = r'.\dataset\sim_struct'  # path to the 'sim_struct' directory

batch_size = 4

# Optimizer parameter
model_lr = 1e-3
model_betas = (0.9, 0.99)
model_eps = 1e-8
model_weight_decay = 1e-5  # weight decay (e.g., 1e-4 or 1e-5) can be beneficial as it adds L2 regularization.

# EMA parameter
model_ema_decay = 0.5
# How many iterations to print the training result
print_frequency = 100
train_print_frequency = 5
valid_print_frequency = 10

# Dynamically adjust the learning rate policy
lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
lr_scheduler_gamma = 0.1

exp_name = "autoencoder"
device = "cpu"
