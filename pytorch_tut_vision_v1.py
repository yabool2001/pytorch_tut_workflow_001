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

class CircleModelV3 ( nn.Module ) :
# Construct a model (oryginal V2) class that subclasses nn.Module with non-linear activation function
# Jest lepsze niż nn.Sequential w przypadkach bardziej skomplikowanych struktur
# Oryginal version comprises 3 hidden layers and 10 features
    def __init__ ( self ):
        super ().__init__ ()
        # Create 3 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        # Create 2 nn.Linear layers capable of handling X and y input and output shapes
        # self.layer_1 = nn.Linear ( in_features =  2 , out_features = 3 ) # takes in 2 features (X), produces 3 features
        # self.layer_2 = nn.Linear ( in_features = 3 , out_features = 1 ) # takes in 3 features (X), produces 10 features
        # self.layer_3 = nn.Linear ( in_features = 10 , out_features =  1 ) # takes in 10 features, produces 1 feature (y)
        self.relu = nn.ReLU () # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()
    
    # 3. Define a forward method containing the forward pass computation
    def forward ( self , x ) :
        # Intersperse the ReLU activation function between layers
        return self.layer_3 ( self.relu ( self.layer_2 ( self.relu ( self.layer_1 ( x ) ) ) ) )
        # return self.layer_2 ( self.relu ( self.layer_1 ( x ) ) )

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



# Create an instance of the model and send it to target device
model_3 = CircleModelV3 ().to ( 'cpu' )
# print ( f"\n{model_3 = }" )
# print ( f"\n{model_3.state_dict () = }")

