from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2

original = cv2.imread("images/test_image_og.jpg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

altered = cv2.imread("images/test_image_og_contrast.jpg")
altered = cv2.cvtColor(altered, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(original, altered, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

#find the contours
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#draw rectangles
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.rectangle(altered, (x,y), (x+w, y+h), (0,0,255), 2)
    
#show output
cv2.imshow("Original", original)
cv2.imshow("Altered", altered)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

