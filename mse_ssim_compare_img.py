from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import cv2

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print(err)
    return err

def compare_images(imageA, imageB, title):
        m = mse(imageA, imageB)
        s = ssim(imageA, imageB)
        
        print(s)
        
        fig = plt.figure(title)
        plt.suptitle("MSE : %.2f, SSIM: %.2f" % (m, s))
        
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")
        
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")
        
        plt.show()
        
original = cv2.imread("images/test_image_og.jpg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

darker = cv2.imread("images/test_image_dark.jpg")
darker = cv2.cvtColor(darker, cv2.COLOR_BGR2GRAY)

brushed = cv2.imread("images/test_image_brush.jpg")
brushed = cv2.cvtColor(brushed, cv2.COLOR_BGR2GRAY)

fig = plt.figure("Images")
images = ("Original", original), ("Darker", darker), ("Brushed", brushed)

for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")

plt.show()

compare_images(original, original, "1 to 1")
compare_images(original, darker, "OG vs darker")
compare_images(original, brushed, "OG vs brushed")
compare_images(darker, brushed, "darker vs brushed")
