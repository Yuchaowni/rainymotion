"""
    File name: save.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-04-01
    Python Version: 3.6
"""


def binary(data, filename, dtype):
    """
    Save data as binary file
    :param data: np.ndarray, data that need to ve saved
    :param filename: str, output file name
    :param dtype: str, data type
    :return: None
    """
    data = data.astype(dtype)
    with open(filename, 'wb') as f:
        data.tofile(f)

