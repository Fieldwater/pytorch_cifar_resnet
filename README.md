# pytorch_cifar_resnet

## How to load model parameters

```
# load model
import torch
# from project1_model import project1_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = project1_model().to(device)
model = torch.nn.DataParallel(model)  # speed up
model_path = "./project1_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
```


## How to reproduce the model
There are two approaches .ipynb and .py file to train and reproduce our model.

1. Directly use main.py to train model in the python environment with torch tools
```
python main.py
```
2. Based on jupyter notebook, you can intuitively train and observe the intermediate results.
