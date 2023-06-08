# Rozgrzebane
# 2023.06.08
# Source: https://www.learnpytorch.io/03_pytorch_computer_vision/
# Computer vision is the art of teaching a computer to see.
# Dependencies: saved_model/pytorch_tut_.pth

from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

model_saved_path = Path ( "saved_model/pytorch_tut_classification_V4.pth" )
new_training = 0
train_epochs = 100000

def accuracy_fn ( y_true , y_pred ) :
    correct = torch.eq ( y_true , y_pred ).sum ().item ()
    acc = (correct / len ( y_pred ) ) * 100
    return acc

def test_model ( model , X_test , y_test , loss_fn ) :
    model.eval ()
    with torch.inference_mode () :
        # 1. Forward pass
        test_logits = model ( X_test ).squeeze ()
        test_pred = torch.round ( torch.sigmoid ( test_logits ) ) # logits -> prediction probabilities -> prediction labels
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn ( test_logits , y_test )
        test_acc = accuracy_fn ( y_true = y_test , y_pred = test_pred )
        print ( f"Test Loss: {test_loss:.5f} , Test Accuracy: {test_acc:.2f}%" )
    return test_pred

def train_model ( model , X_train , loss_fn ) :
    # get_model_parameters ( model , 'before train' )
    optimizer = torch.optim.SGD ( params = model.parameters () , lr = 0.001 )
    # Create empty loss lists to track values
    model.train ()
    for epoch in range ( train_epochs ) :
        # 1. Forward pass
        y_logits = model ( X_train ).squeeze ()
        y_pred = torch.round ( torch.sigmoid ( y_logits ) ) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn ( y_logits , y_train ) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn ( y_true = y_train , y_pred = y_pred )
        
        # 3. Optimizer zero grad
        optimizer.zero_grad ()
        
        # 4. Loss backward
        loss.backward ()

         # 5. Optimizer step
        optimizer.step ()

        # Check
        if epoch % ( round ( train_epochs / 10 ) ) == 0 :
            print ( f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}%" )

# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# #############
# ### START APP
# #############

# Check versions
print ( f"PyTorch version: {torch.__version__}, torchvision version: {torchvision.__version__}" )

# Setup training data
train_data = datasets.FashionMNIST (
    root = "" , # where to download data to?
    train = True , # get training data
    download = True , # download data if it doesn't exist on disk
    transform = ToTensor() , # images come as PIL format, we want to turn into Torch tensors
    target_transform = None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST (
    root = "" ,
    train = False , # get test data
    download = True ,
    transform = ToTensor()
)

# See first training sample
print ( f"\n{train_data.data.shape = }")
image , label = train_data [0]
print ( f"\n{image = }" )
print ( f"\n{image.shape = }")
print ( f"\n{label = }" )
print ( f"\n{train_data.classes = }")

# Visualizing sample from Dataset
image, label = train_data[10]
plt.imshow ( image.squeeze() ) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title ( train_data.classes[label] + " " + str ( label ) )
plt.show ()
### Create DataLoader's for our training and test sets.

# Setup the batch size hyperparameter
BATCH_SIZE = 32
# Turn datasets into iterables (batches)
train_dataloader = DataLoader ( train_data , # dataset to turn into iterable
    batch_size = BATCH_SIZE , # how many samples per batch? 
    shuffle = True # shuffle data every epoch?
)
test_dataloader = DataLoader ( test_data ,
    batch_size = BATCH_SIZE ,
    shuffle = False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
print(f"{train_dataloader =}, {test_dataloader =}") 
print(f"Length of train dataloader: {len ( train_dataloader ) } batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len ( test_dataloader ) } batches of {BATCH_SIZE}")

# Visualizing sample from DataLoader
image, label = train_data[10]
plt.imshow ( image.squeeze() ) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title ( train_data.classes[label] + " " + str ( label ) )
plt.show ()

# Create an instance of the model and send it to target device
torch.manual_seed(42)
model_2 = FashionMNISTModelV2 ( input_shape = 1 , hidden_units = 10 , output_shape = len ( train_data.classes ) )
# print ( f"\n{model_3 = }" )
# print ( f"\n{model_3.state_dict () = }")

