import numpy as np

def read_sunspot_data(filename):
    """
    Load sunspot data from a text file.
    Expects each line to contain several columns of data,
    with the 4th column (index 3) being the target value.

    :param filename: Name of the input file.
    :return: A NumPy array of values extracted from the 4th column.
    """
    values = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                val = float(parts[3])
                values.append(val)
    return np.array(values)
