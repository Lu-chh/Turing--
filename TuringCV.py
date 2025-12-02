import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.font_manager import FontProperties


# 画五角星
def draw_star(canvas, center, r1, r2, color, thickness):
    cx, cy = center
    points = []
    for i in range(10):                    #判断内角还是外角
        r = r1 if i % 2 == 0 else r2
        angle = np.deg2rad(i * 36 - 90)
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append([x, y])
    points = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], True, color, thickness)

canvas = np.ones((800, 800, 3), np.uint8) * 255

#画矩形
cv2.rectangle(canvas, (10, 10), (100, 200), (0, 0, 255), 1)
cv2.rectangle(canvas, (120, 20), (500, 200), (255, 0, 0), 1)
cv2.rectangle(canvas, (300, 300), (400, 400), (0, 255, 0), -1)

#画圆
cv2.circle(canvas, (650, 400), 100, (0, 200, 200), 2)
cv2.circle(canvas, (100, 600), 70, (255, 0, 255), -1)

#画三角形
triangle1 = np.array([[150, 100], [250, 50], [300, 150]], np.int32).reshape(-1, 1, 2)
cv2.polylines(canvas, [triangle1], True, (0, 150, 200), 1)
triangle2 = np.array([[550, 150], [650, 100], [700, 250]], np.int32).reshape(-1, 1, 2)
cv2.polylines(canvas, [triangle2], True, (128, 0, 128), 1)

#画椭圆
src_points1 = np.float32([[0, 100], [200, 100], [100, 200]])      #仿射变换1
dst_points1 = np.float32([[0, 100], [200, 100], [100, 150]])
M1 = cv2.getAffineTransform(src_points1, dst_points1)

src_points2 = np.float32([[0, 100], [200, 100], [100, 200]])      #仿射变换2
dst_points2 = np.float32([[0, 100], [100, 100], [100, 200]])
M2 = cv2.getAffineTransform(src_points2, dst_points2)

oval1 = []
for angle in np.linspace(0, 2*np.pi, 600):
    x = 200 + 80 * np.cos(angle)
    y = 500 + 80 * np.sin(angle)
    pt = np.array([x, y, 1], dtype=np.float32)
    new_pt = M1 @ pt
    oval1.append(new_pt)
ellipse_points = np.array(oval1, np.int32).reshape(-1, 1, 2)
cv2.polylines(canvas, [ellipse_points], True, (255, 255, 0), 1)

oval2 = []
for angle in np.linspace(0, 2*np.pi, 600):
    x = 700 + 80 * np.cos(angle)
    y = 700 + 80 * np.sin(angle)
    pt = np.array([x, y, 1], dtype=np.float32)
    new_pt = M2 @ pt
    oval2.append(new_pt)
ellipse_points = np.array(oval2, np.int32).reshape(-1, 1, 2)
cv2.polylines(canvas, [ellipse_points], True, (114, 51, 4), 1)

#画五角星
draw_star(canvas,(400, 625),200,80,(0, 0, 255),1)

#截取模板
template = canvas[10:201, 10:101]
cv2.imwrite("template.png", template)

#赋值画板
matching = canvas.copy()
contours = canvas.copy()

# 预处理模板并获得模板的轮廓
template_img = cv2.imread("template.png", 0)
_, template_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
template_cnts, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template_cnt = template_cnts[0]

#预处理画板
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

#闭运算 处理椭圆断裂
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#获取轮廓
cnts, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #部分轮廓存在两层 需要筛选内外层

#存放不重复部分
true_cnts = []

#对每个轮廓进行筛选和判断
for i, h in enumerate(hierarchy[0]):
    parent = h[3]
    if parent == -1 or parent == 15:
        true_cnts.append(cnts[i])

        #获取每个轮廓的近似
        epsilon = 0.02 * cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], epsilon, True)

        #获取圆度
        area = cv2.contourArea(cnts[i])
        perimeter = cv2.arcLength(cnts[i], True)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        match_score = cv2.matchShapes(cnts[i], template_cnt, cv2.CONTOURS_MATCH_I1, 0)

        #判断是否与模板匹配
        if match_score > 0.2:
            x, y, w, h = cv2.boundingRect(cnts[i])
            cv2.rectangle(matching, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cx = x + w // 2
            cy = y + h // 2

            text = f"{match_score:.2f}"
            position = (cx, cy)
            color = (0, 0, 0)
            font_scale = 1
            thickness = 2

            cv2.putText(matching, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        #判断每个轮廓具体种类
        if len(approx) == 3:
            shape_type = "triangle"
        elif len(approx) == 4:
            shape_type = "rectangle"
        elif len(approx) == 10:
            shape_type = "pentagram"
        elif len(approx) > 4 and circularity > 0.8:
            shape_type = "circle"
        elif len(approx) > 4 and circularity <= 0.8:
            shape_type = "ellipse"

        x, y, w, h = cv2.boundingRect(cnts[i])
        cv2.rectangle(contours, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        text = shape_type
        position = (cx, cy)
        color = (0, 0, 0)
        font_scale = 1
        thickness = 2

        cv2.putText(contours, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

print("图形数量:", len(true_cnts))

plt.imshow(canvas)
plt.show()

cv2.imwrite("contours.png", contours)
cv2.imwrite("matching.png", matching)
cv2.imwrite("shapes.png", canvas)
cv2.imwrite("edges.png", edges)
