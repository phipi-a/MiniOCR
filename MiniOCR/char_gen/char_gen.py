import PIL.Image
import numpy as np
import torch
import os

digit_register = {
    "0": 0b01111110,
    "1": 0b00110000,
    "2": 0b01101101,
    "3": 0b01111001,
    "4": 0b00110011,
    "5": 0b01011011,
    "6": 0b01011111,
    "7": 0b01110000,
    "8": 0b01111111,
    "9": 0b01111011,
    "A": 0b01110111,
    "B": 0b00011111,
    "C": 0b01001110,
    "D": 0b00111101,
    "E": 0b01001111,
    "F": 0b01000111,
    "G": 0b01011110,
    "H": 0b00110111,
    "I": 0b00000110,
    "J": 0b00111100,
    "K": 0b01010111,
    "L": 0b00001110,
    "M": 0b01010100,
    "N": 0b01110110,
    "O": 0b01111110,
    "P": 0b01100111,
    "Q": 0b01101011,
    "R": 0b01100110,
    "S": 0b01011011,
    "T": 0b00001111,
    "U": 0b00111110,
    "V": 0b00111110,
    "W": 0b00101010,
    "X": 0b00110111,
    "Y": 0b00111011,
    "Z": 0b01101101,
    "a": 0b01111101,
    "b": 0b00011111,
    "c": 0b00001101,
    "d": 0b00111101,
    "e": 0b01101111,
    "f": 0b01000111,
    "g": 0b01111011,
    "h": 0b00010111,
    "i": 0b00000100,
    "j": 0b00011000,
    "k": 0b01010111,
    "l": 0b00000110,
    "m": 0b00010100,
    "n": 0b00010101,
    "o": 0b00011101,
    "p": 0b01100111,
    "q": 0b01110011,
    "r": 0b00000101,
    "s": 0b01011011,
    "t": 0b00001111,
    "u": 0b00011100,
    "v": 0b00011100,
    "w": 0b00010100,
    "x": 0b00110111,
    "y": 0b00111011,
    "z": 0b01101101,
}

# get curent file path
current_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_path, "imgs")
ve = PIL.Image.open(os.path.join(image_path, "ve.png"))
ho = PIL.Image.open(os.path.join(image_path, "ho.png"))


def get_char_image(char):
    image_dict = [ho, ve, ve, ho, ve, ve, ho]
    pos_center_dict = [
        (45, 21),
        (74, 53),
        (74, 119),
        (45, 151),
        (16, 119),
        (16, 53),
        (45, 81),
    ]

    def insert_elem(pos, img):
        insert_image = image_dict[pos]
        insert_center = pos_center_dict[pos]
        img.paste(
            insert_image,
            (
                insert_center[0] - int(insert_image.size[0] / 2),
                insert_center[1] - int(insert_image.size[1] / 2),
            ),
            insert_image,
        )
        return img

    c_de = digit_register[char]
    # get bin as string
    bin_str = format(c_de, "07b")
    image_height = 172
    image_width = 90
    image = PIL.Image.new("RGB", (image_width, image_height), "black")
    for i in range(7):
        if bin_str[i] == "1":
            image = insert_elem(i, image)
    # add border of 10 pixels
    image = PIL.ImageOps.expand(image, border=10, fill="black")
    return image


def get_all_char_images(characters=None):
    images = []
    chars = []
    if characters is None:
        characters = digit_register.keys()
    for char in characters:
        img = get_char_image(char)
        images.append(np.array(img))
        chars.append(char)
    np_images = np.stack(images)
    torch_images = torch.tensor(np_images)
    torch_images = torch.any(torch_images, dim=-1).float()
    return torch_images, chars
