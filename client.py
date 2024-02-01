import math
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tritonclient.http as httpclient
from preprocess import *
from postprocess import *
from configs import *
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

SAVE_INTERMEDIATE_IMAGES = False

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.5
    iou_thres = 0.3

cfg = CFG()

def visualize_image(image, predictions, save_path=None, conf_threshold=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = Image.fromarray(image)  # Convert NumPy array to PIL Image
    draw = ImageDraw.Draw(image_copy)

    for pred in predictions:
        x1, y1, x2, y2, conf, id = pred
        if conf >= conf_threshold:
            box = [int(x1), int(y1), int(x2), int(y2)]
            id = int(id)
            draw.rectangle(box, outline=IDX2COLORs[id], width=2)
            draw.text((box[0], box[1]), f"{IDX2TAGs[id]} ({conf:.2f}%)", fill=IDX2COLORs[id])

    plt.imshow(image_copy)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Open a file dialog to select an image
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        exit()

    raw_image = cv2.imread(image_path)

    resized_pad_image, ratio, (padd_left, padd_top) = resize_and_pad(raw_image, new_shape=cfg.image_size)
    norm_image = normalization_input(resized_pad_image)


    detection_input = httpclient.InferInput(
        "images", norm_image.shape, datatype="FP32"
    )

    detection_input.set_data_from_numpy(norm_image, binary_data=True)

    detection_response = client.infer(
        model_name="traffic", inputs=[detection_input]
    )
    result = detection_response.as_numpy("output0")

    pred = postprocess(result, cfg.conf_thres, cfg.iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio

    # Visualize the image with bounding box predictions
    visualize_image(raw_image, pred)

    # Optionally, save the intermediate image
    if SAVE_INTERMEDIATE_IMAGES:
        intermediate_image_path = "./intermediate_result.jpg"
        cv2.imwrite(intermediate_image_path, resized_pad_image)
        print(f"Intermediate image saved at: {intermediate_image_path}")
