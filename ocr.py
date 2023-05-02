import numpy as np
import cv2
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("output/model.pth").to(device)
model.eval()

# image = image.to(device)
# pred = model(image)
# idx = pred.argmax(axis=1).cpu().numpy()[0]


image = cv2.imread("images/image3.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 125, 255, 0)
thresh = cv2.bitwise_not(thresh)

# cv2.imshow("thresh", thresh)

# kernel = np.ones((5,5), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow("MORPH_OPEN", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(hierarchy)

areas = []
bounding_boxes = []
wh_ratios = []
for cnt in contours:
    areas.append(cv2.contourArea(cnt))
    x,y,w,h = cv2.boundingRect(cnt)
    bounding_boxes.append([x,y,w,h])
    wh_ratios.append(w/h)
area_mean = np.mean(areas)
wh_ratio_mean = np.mean(wh_ratios)
print("area mean:", area_mean)
print("wh ratio mean:", wh_ratio_mean)

for i, cnt in enumerate(contours):
    # check if contour doesn't have parrent (is not inside another contour)
    if hierarchy[0][i][3] == -1:
        # check if box is too small
        area = areas[i]
        if area < area_mean * 0.2:
            continue
        
        # draw bounding box
        x,y,w,h = bounding_boxes[i]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        
        # max = np.max((w,h))
        # img = np.zeros((max, max))
        # img[(max-h)//2:h + (max-h)//2, (max-w)//2:w + (max-w)//2] = thresh[y:y+h, x:x+w]
        # img = cv2.resize(img, (28, 28)).reshape([1, 1, 28, 28])
        # img = torch.tensor(img).to(device=device, dtype=torch.float32)
        # with torch.no_grad():
        #     pred = model(img)
        # idx = pred.argmax(axis=1).cpu().numpy()[0]

        # cv2.putText(image, str(idx), (x, y),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 2)
        
        nums = 1
        if area > area_mean * 2:
            wh_ratio = wh_ratios[i]
            nums = int(np.ceil(wh_ratio / wh_ratio_mean))
            for j in range(nums):
                cv2.rectangle(image,(int(x + w*j/nums),y),(int(x+w*(j+1)/nums),y+h),(0,0,255),2)
        
        for j in range(nums):
            new_w = int(w/nums)
            max = np.max((new_w,h))
            img = np.zeros((max, max))
            try:
                img[(max-h)//2:h + (max-h)//2, (max-new_w)//2:new_w + (max-new_w)//2] = thresh[y:y+h, int(x + w*j/nums):int(x+w*(j+1)/nums)]
            except Exception:
                img[(max-h)//2:h + (max-h)//2, (max-new_w)//2:new_w + (max-new_w)//2 + 1] = thresh[y:y+h, int(x + w*j/nums):int(x+w*(j+1)/nums)]
            img = cv2.resize(img, (28, 28)).reshape([1, 1, 28, 28])
            img = torch.tensor(img).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = model(img)
            idx = pred.argmax(axis=1).cpu().numpy()[0]

            cv2.putText(image, str(idx), (int(x + w*j/nums), y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 2)

        # draw contour
        # cv2.drawContours(image, [cnt], -1, (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), 3)

cv2.imshow("kir", image)
cv2.waitKey()