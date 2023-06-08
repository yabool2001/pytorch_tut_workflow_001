# 2023.05.29
# Source: https://www.learnpytorch.io/02_pytorch_classification/
# A classification problem involves predicting whether something is one thing or another.
# Dependencies: saved_model/pytorch_tut_classification_V3.pth

from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

model_saved_path = Path ( "saved_model/pytorch_tut_classification_V3.pth" )
train_epochs = 2500000
new_training = 2
mode = 0 # 0: always train and test, 1: train if not saved and test, 2: newer train and always test

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to ("cpu")
    X, y = X.to ("cpu"), y.to ("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show ()

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
        self.layer_1 = nn.Linear ( in_features = 2 , out_features = 10 )
        self.layer_2 = nn.Linear ( in_features = 10 , out_features = 10 )
        self.layer_3 = nn.Linear ( in_features = 10 , out_features = 1 )
        # Create 2 nn.Linear layers capable of handling X and y input and output shapes
        #self.layer_1 = nn.Linear ( in_features =  2 , out_features = 3 ) # takes in 2 features (X), produces 3 features
        #self.layer_2 = nn.Linear ( in_features = 3 , out_features = 1 ) # takes in 3 features (X), produces 10 features
        self.relu = nn.ReLU () # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()
    
    # 3. Define a forward method containing the forward pass computation
    def forward ( self , x ) :
        # Intersperse the ReLU activation function between layers
        return self.layer_3 ( self.relu ( self.layer_2 ( self.relu ( self.layer_1 ( x ) ) ) ) )
        #return self.layer_2 ( self.relu ( self.layer_1 ( x ) ) )

# #############
# ### START APP
# #############

# Create the data
X , y = make_circles ( n_samples = 1000 , noise = 0.03 , random_state = 42 )
X2 , y2 = make_circles ( n_samples = 1000 , noise = 0.3 )
# Visualize the data
# Na wykresie weź kolumnę 0 jako x, a kolumnę 1 jako y:
# plt.scatter ( x = X[ : , 0 ], y = X[ : , 1 ] , c = y, cmap = plt.cm.RdBu )
# plt.show ()

X = torch.from_numpy ( X ).type ( torch.float )
y = torch.from_numpy ( y ).type ( torch.float )
X2 = torch.from_numpy ( X2 ).type ( torch.float )
y2 = torch.from_numpy ( y2 ).type ( torch.float )
print ( f"\n{X[ :5 ] = }" ) , print ( f"{X.shape = }" )
print ( f"\n{y[ :5 ] = }" ) , print ( f"{y.shape = }" )

# test_size: 20% test, 80% train
# random_state: make the random split reproducible
X_train , X_test , y_train , y_test = train_test_split ( X , y , test_size = 0.2 , random_state = 42 )
#X_train , X_test , y_train , y_test = train_test_split ( X2 , y2 , test_size = 0.2 )

# Create an instance of the model and send it to target device
model_3 = CircleModelV3 ().to ( 'cpu' )

print ( f"\n{model_3 = }" )
print ( f"\n{model_3.state_dict () = }")


loss_fn = nn.BCEWithLogitsLoss ()
optimizer = torch.optim.SGD  ( params = model_3.parameters () , lr = 0.1 )

if model_saved_path.is_file () or mode > 0 :
    model_3.load_state_dict ( torch.load ( f = model_saved_path ) ) # Załaduj istniejący model
else :
    train_model ( model_3 , X_train , loss_fn ) # Trenuj model, bo nie ma zapisanego
    new_training = 1
if new_training :
    print (f"Saving model to: {model_saved_path}")
    torch.save ( obj = model_3.state_dict () , f = model_saved_path )

test_model ( model_3 , X_test , y_test , loss_fn ) # Trenuj model, bo nie ma zapisanego

#plt.subplot ( 1 , 2 , 2 )
plt.title ( "Test" )
plot_decision_boundary ( model_3 , X_test , y_test ) # model_3 = has non-linearity