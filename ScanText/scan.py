# 导入工具包
import numpy as np
import argparse
import cv2
import json
import os

# 加载配置文件
def load_config(config_path="settings.json"):
    """
    加载配置文件，如果文件不存在则使用默认配置
    配置文件路径相对于scan.py文件所在目录
    """
    default_config = {
        "image_path": "path/to/your/image.jpg",
        "output_path": "scan.jpg",
        "canny_thresholds": {"lower": 75, "upper": 200},
        "gaussian_blur": {"kernel_size": [5, 5], "sigma": 0},
        "resize_height": 500,
        "approx_epsilon_ratio": 0.02,
        "binary_threshold": 100
    }
    
    # 获取scan.py文件所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果config_path不是绝对路径，则相对于脚本目录
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合并默认配置，确保所有必要的键都存在
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                        
                # 处理相对路径的image_path
                if not os.path.isabs(config["image_path"]):
                    config["image_path"] = os.path.join(script_dir, config["image_path"])
                    
                # 处理相对路径的output_path
                if not os.path.isabs(config["output_path"]):
                    config["output_path"] = os.path.join(script_dir, config["output_path"])
                    
                print(f"成功加载配置文件: {config_path}")
                return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"配置文件读取失败: {e}，使用默认配置")
            return default_config
    else:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return default_config

# 设置参数 - 支持命令行参数和配置文件
ap = argparse.ArgumentParser(description="文档扫描工具")
ap.add_argument("-i", "--image", 
                help="待扫描图片路径 (可选，优先级高于配置文件)")
ap.add_argument("-c", "--config", default="settings.json",
                help="配置文件路径 (默认: settings.json)")
args = vars(ap.parse_args())

# 加载配置
config = load_config(args["config"])

# 确定图片路径：命令行参数优先，否则使用配置文件
image_path = args["image"] if args["image"] else config["image_path"]

# 处理命令行参数的相对路径
if args["image"] and not os.path.isabs(args["image"]):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, args["image"])

# 验证图片路径
if not os.path.exists(image_path):
    print(f"错误: 图片文件不存在: {image_path}")
    print("请检查配置文件中的image_path或使用-i参数指定正确的图片路径")
    exit(1)

def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype = "float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 读取输入
image = cv2.imread(image_path)
if image is None:
    print(f"错误: 无法读取图片: {image_path}")
    exit(1)
    
print(f"正在处理图片: {image_path}")

#坐标也会相同变化
ratio = image.shape[0] / config["resize_height"]
orig = image.copy()

image = resize(orig, height=config["resize_height"])

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, tuple(config["gaussian_blur"]["kernel_size"]), 
                       config["gaussian_blur"]["sigma"])
edged = cv2.Canny(gray, config["canny_thresholds"]["lower"], 
                  config["canny_thresholds"]["upper"])

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测 - 兼容OpenCV 3.x和4.x版本
contours_result = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# OpenCV 4.x 返回 (contours, hierarchy)，OpenCV 3.x 返回 (image, contours, hierarchy)
if len(contours_result) == 2:
    cnts, hierarchy = contours_result  # OpenCV 4.x
else:
    _, cnts, hierarchy = contours_result  # OpenCV 3.x

print(f"检测到 {len(cnts)} 个原始轮廓")

# 检查是否找到轮廓
if len(cnts) == 0:
    print("错误: 未找到任何轮廓，请检查图片质量或调整Canny阈值")
    exit(1)

# 过滤掉面积过小的轮廓，避免计算错误
valid_cnts = []
for i, cnt in enumerate(cnts):
    try:
        # 验证轮廓数据格式
        if cnt is None or len(cnt) < 3:
            print(f"跳过无效轮廓 {i+1}: 点数不足")
            continue
            
        # 检查轮廓数据类型和形状
        if not isinstance(cnt, np.ndarray):
            print(f"跳过无效轮廓 {i+1}: 数据类型错误")
            continue
            
        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        print(f"检查轮廓 {i+1}: 点数={len(cnt)}, 面积={area:.0f}")
        
        # 过滤大小适中的轮廓
        if 100 < area < (image.shape[0] * image.shape[1] * 0.9):  # 面积在100到图片面积90%之间
            valid_cnts.append(cnt)
            print(f"✓ 轮廓 {i+1} 通过验证")
        else:
            print(f"跳过轮廓 {i+1}: 面积不合适 ({area:.0f})")
            
    except Exception as e:
        print(f"错误: 无效轮廓 {i+1}: {e}")
        continue  # 跳过无效轮廓

if len(valid_cnts) == 0:
    print("错误: 未找到有效轮廓，请检查图片质量")
    exit(1)

cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:5]
print(f"找到 {len(cnts)} 个有效轮廓")

# 遍历轮廓
screenCnt = None
for i, c in enumerate(cnts):
    # 计算轮廓近似
    # peri 表示轮廓周长
    peri = cv2.arcLength(c, True)
    # C表示输入的点集
    # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    # True表示封闭的
    approx = cv2.approxPolyDP(c, config["approx_epsilon_ratio"] * peri, True)
    
    print(f"轮廓 {i+1}: 原始点数={len(c)}, 近似点数={len(approx)}, 面积={cv2.contourArea(c):.0f}")

    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        print(f"找到四边形边界: 轮廓 {i+1}")
        break

if screenCnt is None:
    print("警告: 未找到四边形文档边界，将使用最大轮廓")
    # 使用最大轮廓作为备选
    if len(cnts) > 0:
        largest_contour = cnts[0]
        peri = cv2.arcLength(largest_contour, True)
        # 放宽精度，尝试获得四边形
        for epsilon_ratio in [0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(largest_contour, epsilon_ratio * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                print(f"使用放宽精度 {epsilon_ratio} 找到四边形")
                break
                
if screenCnt is None:
    print("错误: 无法找到有效的文档边界，请检查图片质量")
    exit(1)

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, config["binary_threshold"], 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(config["output_path"], ref)
print(f"扫描结果已保存到: {config['output_path']}")
# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(ref, height = 650))
cv2.waitKey(0)