import cv2
import socket
import numpy as np

host = socket.gethostname()
port = 5000

client_socket = socket.socket()
client_socket.connect((host, port))

def callback(event, x, y, flags, params):
  global mouse_x, mouse_y
  if event == cv2.EVENT_LBUTTONDBLCLK:
    mouse_x, mouse_y = x, y
    msg = " ".join(list(map(str, params[y, x, 1:].tolist())))
    client_socket.send(msg.encode())

def click_image(img_path, ann_path):
  # open image
  img = cv2.imread(img_path)
  # open annotations
  ann = np.load(ann_path)

  cv2.namedWindow('image')
  cv2.setMouseCallback('image', callback, ann)

  while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
      break
  client_socket.close()


click_image("big_000420.png", "a000420.npy")


"""
message = input(" -> ")

while message.lower().strip() != 'bye':
    client_socket.send(message.encode())
    message = input(" -> ")

"""

# (-0.55251166, 0.412381, -0.08782556)
