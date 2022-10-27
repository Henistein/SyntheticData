import unittest
import numpy as np
from PIL import Image
import cv2

class TestStringMethods(unittest.TestCase):
  def test_foo(self):
    data = np.load('faces_pixels1.npy', allow_pickle=True)
    
    # create image
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    for face,pixels in data:
      color = np.random.randint(0, 255, size=(3,)).tolist()
      pixels = np.array(list(map(list, pixels)), dtype=np.int32).reshape(-1, 1, 2)
      image = cv2.polylines(image, [pixels], True, color, 1)

    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
  unittest.main()
