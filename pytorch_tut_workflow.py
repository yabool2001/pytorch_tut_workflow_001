# 2023.05.26
# Source: https://www.learnpytorch.io/01_pytorch_workflow/
# A PyTorch model that learns the pattern of the straight line and matches it. 
# Dependencies: saved_model/pytorch_tut_workflow.pth

from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

model_saved_path = Path ( "saved_model/pytorch_tut_workflow.pth" )


def get_model_parameters ( model , comment ) :
    print ( f"{model._get_name()} model parameters {comment}: bias: {model.bias[0]:4.4f}, weights: {model.weights[0]:4.4f}" )

def plt_visualization ( x , y ) :
    #plt.xlim ( -1 , 1 )
    #plt.ylim ( -1 , 1 )
    plt.scatter ( x , y , c = "b" , s = 2 ) # rozpraszać
    plt.pause ( 0.001 )
    plt.show ()
def plot_predictions ( train_data , train_labels , test_data , test_labels , predictions = None ) :
    # Plots training data, test data and compares predictions.
    plt.figure ( figsize = ( 10 , 7 ) )
    # Plot training data in blue
    plt.scatter ( train_data , train_labels , c = "b" , s = 5 , label = f"Training data - length: {len ( train_data )}" )
    # Plot test data in green
    plt.scatter ( test_data , test_labels , c = "g" , s = 5 , label = f"Testing data - length: {len ( train_labels )}" )
    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter ( test_data , predictions , c = "r" , s = 4 , label = "Predictions" )
    # Show the legend
    plt.legend ( prop = { "size" : 14 } )
    plt.show ()

# Parameters to be discovered:
weight = 2
bias = 1

# Create train parameters:
X_train = torch.tensor ( [ [0.1] , [0.2] , [0.3] ] )
# X_train = torch.tensor ( [ 0.1 , 0.2 , 0.3 ] )
# Create labels:
y_train = weight * X_train + bias
# print ( f"X_train:\n{X_train}\ny_train:\n{y_train}")

# Create test parameters:
X_test = torch.tensor ( [ [0.6] , [0.7] , [0.8] ] )
# X_test = torch.tensor ( [ 0.6 , 0.7 , 0.8 ] )
# Create labels:
y_test = weight * X_test + bias
# print ( f"X_test:\n{X_test[:5]}\ny_test:\n{y_test[:5]}")

def test_model ( model , X_test , loss_fn ) :
    model.eval ()
    with torch.inference_mode () :
        y_pred = model ( X_test )
    test_loss = loss_fn ( y_pred , y_test )
    print ( test_loss )
    return y_pred

def train_model ( model , X_train , loss_fn ) :
    # get_model_parameters ( model , 'before train' )
    optimizer = torch.optim.SGD ( params = model.parameters () , lr = 0.001 )
    epochs = 100000
    # Create empty loss lists to track values
    train_loss_values = []
    train_epochs = []
    model.train ()
    for epoch in range ( epochs ) :
        # model.train ()
        y_pred = model ( X_train ) # Forward pass on train data. To wywołuje metodę forward().
        loss = loss_fn ( y_pred , y_train )
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()
        train_epochs.append ( epoch )
        train_loss_values.append ( loss.detach () )
        # if epoch % ( round ( epochs / 10 ) ) == 0 :
        # if epoch == epochs - 1 :
            # get_model_parameters ( model , ' after train' )
    plt_visualization ( train_epochs , train_loss_values )


# Visualize
# plot_predictions ( X_train , y_train , X_test , y_test )

class LinearRegressionModel ( nn.Module ) :
# A Linear Regression model class
    def __init__ ( self ) :
        super ().__init__ () 
        self.weights = nn.Parameter ( torch.randn ( 1 , dtype = torch.float ) , requires_grad = True )
        self.bias = nn.Parameter ( torch.randn ( 1 , dtype = torch.float ) , requires_grad = True )

    def forward ( self , x: torch.Tensor ) -> torch.Tensor :
        # Overriding forward function is mandatory
        # Forward defines the computation in the model
        # "x" is the input data (e.g. training/testing features)
        # return linear regression formula ( y = m*x + b )
        # Regresja – metoda statystyczna pozwalająca na opisanie współzmienności kilku zmiennych przez dopasowanie do nich funkcji. Umożliwia przewidywanie nieznanych wartości jednych wielkości na podstawie znanych wartości innych. Źródło: https://pl.wikipedia.org/wiki/Regresja_(statystyka)
        return self.weights * x + self.bias

# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed ( 42 )

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel ()
loss_fn = nn.L1Loss () # May be also called Cost Function or Criterion in different areas.


if model_saved_path.is_file () :
    model_0.load_state_dict ( torch.load ( f = model_saved_path ) ) # Załaduj istniejący model
else :
    train_model ( model_0 , X_train , loss_fn ) # Trenuj model, bo nie ma zapisanego
y_pred = test_model  ( model_0 , X_test , loss_fn )
print (f"Saving model to: {model_saved_path}")
torch.save ( obj = model_0.state_dict () , f = model_saved_path )
plot_predictions ( X_train , y_test , X_test , y_test , y_pred )