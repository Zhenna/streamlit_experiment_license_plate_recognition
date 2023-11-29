import torch
import numpy as np


def getBestPrediction(predictions: torch.Tensor) -> np.ndarray:
    """return the biggest bounding box"""

    if len(predictions) == 1:
        return predictions[0, :4].numpy()
    else:
        list_area = []

        for pred_idx in range(len(predictions)):
            x1, y1, x2, y2 = predictions[pred_idx, :4].numpy()
            area = (x2 - x1) * (y2 - y1)
            list_area.append(area)

        best_bb_idx = np.array(list_area).argmax()
        return predictions[best_bb_idx, :4].numpy()



