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
        self.model = template_matching.TemplateMatchingModel(self.imgs)

    def preprocessing(self, img):
        img = torch.any(img, dim=-1).float()
        return img

    def predict(self, img_path):
        img = PIL.Image.open(img_path)
        img = np.array(img)
        img = torch.tensor(img)
        img = self.preprocessing(img).unsqueeze(0)

        res = self.model(img)[0]
        return template_matching.get_top_char_pairs(res, self.chars)

    def predict_batch(self, img_batch):
        img_batch = self.preprocessing(img_batch)
        res = self.model(img_batch)
        out = []
        for r in res:
            out.append(template_matching.get_top_char_pairs(r, self.chars))
        return out
