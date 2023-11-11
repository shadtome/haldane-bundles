import torch

num = 6
device = (f'cuda:{num}'
          if torch.cuda.is_available()
          else 'cpu'
)


