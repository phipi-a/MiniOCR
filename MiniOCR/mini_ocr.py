from MiniOCR.char_gen import char_gen
import PIL
import numpy as np
import torch
from MiniOCR.template_matching import template_matching


class Model:
    def __init__(self, device: str = "cpu", characters: list[str] | None = None):
        self.imgs, self.chars = char_gen.get_all_char_images(characters)
        if device == "cuda":
            self.imgs = self.imgs.cuda()

    def preprocessing(self, img):
        img = torch.any(img, dim=-1).float()
        return img

    def predict(self, img_path):
        img = PIL.Image.open(img_path)
        img = np.array(img)
        img = torch.tensor(img)
        img = self.preprocessing(img)
        res = template_matching.matchTemplateTorch(img, self.imgs)
        return template_matching.get_top_char_pairs(res, self.chars)
