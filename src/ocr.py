import pytesseract
import easyocr
import cv2
import re


class ExtractLicenseNum:
    def __init__(self, path2img: str):
        self.path = path2img

    def textRecognition(self):
        text = pytesseract.image_to_string(
            self.path,
            lang="eng",
            config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        return re.sub(r"\W+", "", text.upper())

    def ocr_easyocr(self):
        image = cv2.imread(self.path)

        reader = easyocr.Reader(["en"], gpu=False)
        detections = reader.readtext(image)

        plate_no = []
        [plate_no.append(line[1]) for line in detections]
        plate_no = "".join(plate_no)

        return re.sub(r"\W+", "", plate_no.upper())