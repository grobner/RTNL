import tensornetwork as tn
from elm_mps import ELM_MPS
import torch

def canonicalize_mps(model : ELM_MPS):
    tn.set_default_backend('pytorch')
    left_matrices = model.mps_layer.left_env.matrices
    right_matrices = model.mps_layer.right_env.matrices

    # convert shape (left_bond_dimension, physical_dimension, right_bond_dimension)
    left_matrices = [left_matrices[i].permute(1, 0, 2).cpu() for i in range(left_matrices.shape[0])]
    right_matrices = [right_matrices[i].permute(2, 0, 1).cpu() for i in range(right_matrices.shape[0])]

    left_vector = model.mps_layer.left_env.vector
    right_vector = model.mps_layer.right_env.vector
    left_vector = torch.reshape(left_vector, (1, -1, 2)).permute(0, 2, 1).cpu()
    right_vector = torch.reshape(right_vector, (1, -1, 2)).permute(1, 2, 0).cpu()

    labeled = model.mps_layer.labeled.tensor.permute(2, 0, 1).cpu()

    list_tensors = [left_vector]
    list_tensors.extend(left_matrices)
    list_tensors.extend([labeled])
    list_tensors.extend(right_matrices)
    list_tensors.extend([right_vector])

    return tn.FiniteMPS(list_tensors)
