from PIL import Image
from pathlib import Path

from src.binarization import binarize
from src.metrics import timing_val


@timing_val
def binarize_test(img_name, output_dir='output'):
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
    img = Image.open(f"input/{img_name}.jpg")
    img.load()
    result = binarize(img, window=75)
    result.save(f"{output_dir}/{img_name}.tif", compression=None)


if __name__ == '__main__':
    imgs = [
        '0001',
        '0025',
        '0026',
        '0027',
        'IMG_0295',
        'IMG_0531',
        'IMG_0532',
        'IMG_0533',
        'IMG_0534',
        'IMG_0535',
        'IMG_0536',
        'IMG_0538',
        'IMG_0539',
        'IMG_0540',
        'IMG_1566',
    ]
    for img in imgs:
        binarize_test(img)
