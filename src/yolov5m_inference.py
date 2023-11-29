import yolov5
import torch


def inference(
    path2img: str,
    show_img: bool = False,
    size_img: int = 640,
    nms_conf_thresh: float = 0.7,
    max_detect: int = 10,
) -> torch.Tensor:
    """Load pre-trained model and make an inference using provided image"""
    model = yolov5.load("keremberke/yolov5m-license-plate")

    model.conf = nms_conf_thresh
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = max_detect

    results = model(path2img, size=size_img)
    results = model(path2img, augment=True)

    if show_img:
        results.show()

    return results.pred[0]
