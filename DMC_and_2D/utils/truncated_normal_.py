import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
  tensor = np.random.randn(*tensor.shape)

  # re sample
  while True:
    cond = np.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
    bound_violations = np.sum(cond)
    if bound_violations == 0:
      break
    tensor[cond] = np.random.randn(bound_violations)
  
  # print('t_n.shape', tensor.shape)
  return tensor