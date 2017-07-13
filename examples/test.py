import cv2

path = 'center_image4.jpg'

center_image = cv2.imread(path)
flipped_image = cv2.flip(center_image, 1)
cv2.imwrite('center_image4_flipped.jpg', flipped_image)