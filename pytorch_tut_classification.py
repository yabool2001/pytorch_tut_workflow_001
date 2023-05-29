# 2023.05.29
# Source: https://www.learnpytorch.io/02_pytorch_classification/
# A classification problem involves predicting whether something is one thing or another.
# pytorch_tut_workflow v2.1.py
# Dependencies: saved_model/pytorch_tut_classification.pth

from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

model_saved_path = Path ( "saved_model/pytorch_tut_classification.pth" )

n_samples = 1000

def print_5_in_out_examples ( x , y ) :
    print ( f"X:\n{x[:5]}\ny:\n{y[:5]}")
class CircleModelV0 ( nn.Module ) :
# Construct a model class that subclasses nn.Module
    def __init__ ( self ):
        super ().__init__ ()
        # Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0 ().to ('cpu')


X , y = make_circles ( n_samples , noise = 0.03 , random_state = 42 )
plt.scatter ( x = X[ : , 0 ], y = X[ : , 1 ] , c = y, cmap = plt.cm.RdYlBu )
# plt.show ()

X = torch.from_numpy ( X ).type ( torch.float )
y = torch.from_numpy ( y ).type ( torch.float )
# print_5_in_out_examples ( X , y )

# test_size: 20% test, 80% train
# random_state: make the random split reproducible
X_train , X_test , y_train , y_test = train_test_split ( X , y , test_size = 0.2 , random_state = 42 )
