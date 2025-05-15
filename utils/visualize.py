import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_video(images, path, texts=None, fps=10):
    if texts is not None:
        assert len(images) == len(texts)
        images = [Image.fromarray(image) for image in images]
        for i, (image, text) in enumerate(zip(images, texts)):
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            draw.text((0, 0), text, (255, 255, 255), font=font)
            images[i] = np.array(image)
    mediapy.write_video(path, images, fps=fps, codec='gif')
