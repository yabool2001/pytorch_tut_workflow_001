# 2023.05.29
# Source: https://www.learnpytorch.io/02_pytorch_classification/
# A classification problem involves predicting whether something is one thing or another.
# pytorch_tut_workflow v2.1.py
# Dependencies: saved_model/pytorch_tut_classification.pth

from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

model_saved_path = Path ( "saved_model/pytorch_tut_workflow.pth" )