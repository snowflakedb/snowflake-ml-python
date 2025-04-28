import json
from typing import Any

import numpy.typing as npt


def convert2D_json_to_3D(array: npt.NDArray[Any]) -> list[list[list[Any]]]:
    final_array = []
    for i in range(array.shape[0]):
        dim_2 = []
        for j in range(array.shape[1]):
            json_to_dict = json.loads(array[i][j])
            dim_3 = []
            for k in range(len(json_to_dict)):
                dim_3.append(float(json_to_dict[str(k)]))
            dim_2.append(dim_3)
        final_array.append(dim_2)
    return final_array
