from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2

empty = cv2.imread("images/utanjati_day_empty.jpeg")
empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)

occupied = cv2.imread("images/utanjati_day_occupied.jpeg")
occupied = cv2.cvtColor(occupied, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(empty, occupied, full=True, window_size=31, k1=0.00001, k2=0.00001)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

#find the contours
#thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#thresh = cv2.adaptiveThreshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
thresh = cv2.adaptiveThreshold(diff,100,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#draw rectangles
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(empty, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.rectangle(occupied, (x,y), (x+w, y+h), (0,0,255), 2)
    
#show output
cv2.imshow("empty", empty)
cv2.imshow("occupied", occupied)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

