from yolov5m_inference import inference
from utils import getBestPrediction
from preprocess import TransformImage
from ocr import ExtractLicenseNum


class run_license_plate_recognition:
    def __init__(self, path2img):
        self.path = path2img

    def getBestBoundingBox(self):
        predictions = inference(
            path2img=self.path,
            show_img=False,
            size_img=640,
            nms_conf_thresh=0.7,
            max_detect=10,
        )

        x1, y1, x2, y2 = getBestPrediction(predictions)
        bbox = x1, y1, x2, y2
        return bbox

    def showBestPrediction(self):
        bbox = self.getBestBoundingBox()
        return TransformImage(self.path).show(bounding_box=bbox, save_img=True)

    def recognize_text(self):
        bbox = self.getBestBoundingBox()
        cropped_img = TransformImage(self.path).crop_ROI(
            bounding_box=bbox, save_img=True
        )
        result = ExtractLicenseNum(cropped_img).ocr_easyocr()
        return result


if __name__ == "__main__":
    DIR2IMG = "raw_images/taxi4.jpeg"

    best_bb = run_license_plate_recognition(DIR2IMG).showBestPrediction()
    text = run_license_plate_recognition(DIR2IMG).recognize_text()
    print(text)
    print(best_bb)
