from typing import Tuple
import cv2
import math
from PIL import Image
from PIL import Image, ImageDraw


class TransformImage:
    def __init__(self, path: str):
        self.path = path

    def show(self, bounding_box: Tuple[float] = None, save_img: bool = False):
        image = Image.open(self.path)
        # w, h = img.size
        if bounding_box:
            x1, y1, x2, y2 = bounding_box
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                ((math.floor(x1), math.floor(y1)), (math.ceil(x2), math.ceil(y2))),
                outline="lime",
                width=5,
            )
        # image.show()
        path2save = "temp/full_img.png"
        if save_img:
            image.save(path2save)
            
        return path2save

    def crop_ROI(self, bounding_box: Tuple[float], save_img: bool = False):
        x1, y1, x2, y2 = bounding_box

        image = cv2.imread(self.path)
        # print("img.shape", image.shape)

        cropped_image = image[
            math.floor(y1) : math.ceil(y2), math.floor(x1) : math.ceil(x2)
        ]

        color_converted = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # pil_image = Image.fromarray(color_converted)
        # pil_image.show()
        path2save = "temp/cropped_img.png"

        if save_img:
            cv2.imwrite(path2save, cropped_image)

        return path2save
