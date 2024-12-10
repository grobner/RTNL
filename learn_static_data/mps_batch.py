from typing import List, Tuple

import numpy as np
import tensornetwork as tn
import torch


def random_initializer(d_phys: int, d_bond: int, std: float = 1e-3,
                       boundary: bool = False) -> np.ndarray:
  """Initializes MPS tensors randomly and close to identity matrices.(from https://github.com/google/TensorNetwork v0.1.0)

  Args:
    d_phys: Physical dimension of MPS.
    d_bond: Bond dimension of MPS.
    std: STD of normal distribution for random initialization.
    boundary: If True returns a tensor of shape (d_phys, d_bond).
      Otherwise returns a tensor of shape (d_phys, d_bond, d_bond).
    Note that d_phys given in this function does not have to be the actual
    MPS physical dimension (eg. it can also be n_labels to initialize there
    label MPS tensor).

  Returns:
    tensor: Random numpy array with shape described above.
  """
  if boundary:
    x = np.zeros((d_phys, d_bond))
    x[:, 0] = 1
  else:
    x = np.array(d_phys * [np.eye(d_bond)])
  x += np.random.normal(0.0, std, size=x.shape)
  return x

class Environment:
    def __init__(self, n_sites: int, d_phys: int,
                d_bond: int, device : torch.device, std: float = 1e-3, dtype=torch.float32):

        self.n_sites, self.dtype = n_sites, dtype
        self.d_phys, self.d_bond = d_phys, d_bond
        self.device = device

        v = random_initializer(self.d_phys, self.d_bond, std=std, boundary=True)
        self.vector = torch.tensor(v, dtype=dtype, device=device)
        matrices = random_initializer(self.d_phys * (n_sites - 1), self.d_bond, std=std)
        self.matrices = torch.tensor(matrices.reshape(n_sites - 1, d_phys, d_bond, d_bond), dtype=dtype, device=device)

    def create_network(self, data: torch.Tensor) -> Tuple[tn.Node]:

        mps_vector = tn.Node(self.vector, axis_names=['in', 'r'])
        mps_matrices = tn.Node(self.matrices, axis_names=['sites', 'in', 'l', 'r'])
        data0_node = tn.Node(data[0], axis_names=['in'])
        data_node = tn.Node(data[1:, :], axis_names=['sites', 'in'])

        return mps_vector, mps_matrices, data0_node, data_node

    def contract_network(self, mps_vector: tn.Node, mps_matrices: tn.Node, data0_node: tn.Node, data_node: tn.Node):
        edge = data0_node['in'] ^ mps_vector['in']

        vector = tn.contract(edge, axis_names=['r'])
        matrices = torch.einsum('si,silr->slr', data_node.tensor, mps_matrices.tensor)

        size = matrices.shape[0]
        tensor = matrices
        while size > 1 :
            half_size = size // 2
            nice_size = 2 * half_size
            leftover = tensor[nice_size:]
            tensor = torch.matmul(tensor[0:nice_size:2], tensor[1:nice_size:2])
            tensor = torch.concat([tensor, leftover], axis=0)
            size = half_size + int(size % 2 == 1)
        node = tn.Node(tensor[0], axis_names=['l', 'r'])
        edge = node['l'] ^ vector['r']

        result = tn.contract(edge, axis_names=['bond'])
        return result

    def predict(self, data: torch.Tensor) -> tn.Node:
        data = data.to(self.device)
        mps_vector, mps_matrices, data0_node, data_node = self.create_network(data)
        return self.contract_network(mps_vector, mps_matrices, data0_node, data_node)



class MatrixProductState:
    def __init__(self,
                 n_sites: int,
                 n_labels: int,
                 d_phys: int,
                 d_bond: int,
                 device,
                 std: float = 1e-3,
                 dtype=torch.float32,
                ):
        self.dtype = dtype
        self.position = n_sites // 2

        l = random_initializer(n_labels, d_bond, std=std)
        labeled = torch.tensor(l, dtype=dtype, device=device)
        self.labeled = tn.Node(labeled, axis_names=['out', 'r', 'l'])

        self.left_env = Environment(self.position, d_phys, d_bond, std=std, device=device)
        self.right_env = Environment(n_sites - self.position - 1, d_phys, d_bond, std=std, device=device)
    def __call__(self, data: torch.Tensor):
        batch_flx = torch.vmap(self.flx)
        return batch_flx(data)

    def flx(self, data: torch.Tensor) -> torch.Tensor:
        left = self.left_env.predict(data[0:self.position, :])
        right = self.right_env.predict(data[self.position:, :])

        left['bond'] ^ self.labeled['l']
        right['bond'] ^ self.labeled['r']
        result = tn.contractors.auto([left, right, self.labeled])
        return result.tensor
