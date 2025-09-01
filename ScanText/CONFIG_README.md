# 配置文件说明文档

## settings.json 参数详解

### 基本配置
- **image_path**: 待扫描的图片文件路径
  - 示例: `"test_image.jpg"`, `"images/document.png"`
  
- **output_path**: 扫描结果保存路径
  - 示例: `"scan_result.jpg"`, `"output/scanned_doc.png"`

### 图像处理参数

#### Canny边缘检测
- **canny_thresholds.lower**: Canny算法低阈值 (推荐: 50-100)
- **canny_thresholds.upper**: Canny算法高阈值 (推荐: 150-250)
  - 阈值越低检测到的边缘越多，越高检测到的边缘越少

#### 高斯模糊
- **gaussian_blur.kernel_size**: 高斯核大小 `[width, height]`
  - 推荐奇数值，如 `[3,3]`, `[5,5]`, `[7,7]`
  - 值越大模糊效果越强，去噪效果越好但也会丢失细节
  
- **gaussian_blur.sigma**: 高斯分布标准差
  - 0表示自动计算，通常保持为0即可

#### 其他参数
- **resize_height**: 预处理时图片缩放高度 (推荐: 400-600)
  - 过小影响检测精度，过大影响处理速度
  
- **approx_epsilon_ratio**: 轮廓近似精度比例 (推荐: 0.01-0.03)
  - 值越小近似越精确，越大近似越粗糙
  - 0.02是经验最佳值
  
- **binary_threshold**: 二值化阈值 (推荐: 80-120)
  - 根据文档对比度调整，文档偏暗用较低值，偏亮用较高值

## 使用方法

### 方法1: 仅使用配置文件
```bash
python scan.py
```

### 方法2: 命令行参数优先
```bash
python scan.py -i "specific_image.jpg"
```

### 方法3: 指定配置文件
```bash
python scan.py -c "custom_settings.json"
```

### 方法4: 组合使用
```bash
python scan.py -c "custom_settings.json" -i "override_image.jpg"
```

## 配置建议

### 不同文档类型的参数调优

#### 清晰文档 (打印件、扫描件)
```json
{
    "canny_thresholds": {"lower": 50, "upper": 150},
    "gaussian_blur": {"kernel_size": [3, 3], "sigma": 0},
    "approx_epsilon_ratio": 0.02,
    "binary_threshold": 100
}
```

#### 模糊文档 (手机拍照)
```json
{
    "canny_thresholds": {"lower": 75, "upper": 200},
    "gaussian_blur": {"kernel_size": [5, 5], "sigma": 0},
    "approx_epsilon_ratio": 0.025,
    "binary_threshold": 90
}
```

#### 低对比度文档
```json
{
    "canny_thresholds": {"lower": 30, "upper": 100},
    "gaussian_blur": {"kernel_size": [3, 3], "sigma": 0},
    "approx_epsilon_ratio": 0.015,
    "binary_threshold": 80
}
```