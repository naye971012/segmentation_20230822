import matplotlib.pyplot as plt
import torch
import numpy as np

num_class = 26

def denoramlize(img):
    img = img.permute(1,2,0)            # change shape ---> (width, height, channel)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    img = img*std + mean
    img = np.clip(img,0,1)              # convert the pixel values range(min=0, max=1)
    return img

def imshow(img, mask):
    fig = plt.figure(figsize=(120, 30))
    a = fig.add_subplot(1, num_class+2 , 1)
    plt.imshow(denoramlize(img), cmap='bone')
    a.set_title("Original image")
    plt.grid(False)
    plt.axis("off")

    for i in range(2,num_class+2 ,1):
      a = fig.add_subplot(1, num_class+2 , i)
      imgplot = plt.imshow(mask[i-2], cmap='binary')
      a.set_title(f"The mask {i-1}")
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])

      plt.axis("off")
      plt.grid(False)