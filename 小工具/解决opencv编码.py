import cv2
import numpy as np
#可以读取带有中文路径的图片地址
def a1():
    def cv_imread(file_path):
        #imdedcode读取的是RGB图像
        cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        return cv_img
    img2 = cv_imread(r'../query\李哓慧\2.jpg')
    cv2.imshow('a',img2)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        raise StopIteration


def a2():
    cv2.namedWindow(winname="image", flags=cv2.WINDOW_AUTOSIZE)  # 或者数字0
    # cv2.resizeWindow("window", 480, 320) #设置图片显示窗口大小


    # 读入图片
    image_path = r"D:\Project\Monet_traffic\data\photo\3.jpg"
    image = cv2.imread(image_path)
    print(image.shape)  # 图片大小
    print(type(image))

    # 改变图片大小,fx代表对图片的长进行缩放尺寸系数
    # image = cv2.resize(image,None,fx=0.5,fy=0.5)


    # 显示图片:窗口名,显示图片
    # cv2.imshow(‘窗口标题’,image)，如果前面没有cv2.namedWindow，自动先执行一个cv2.namedWindow()
    cv2.imshow(winname='image', mat=image)
    # 图片上画框
    cv2.rectangle(img=image, pt1=(285, 40), pt2=(350, 120), color=(0, 255, 0), thickness=2)


    # 添加中文
    # 转换为PIL的image图片格式,使用PIL绘制文字,再转换为OpenCV的图片格式
    def image_add_text(img1, text, left, top, text_color, text_size):
        # 判断图片是否为ndarray格式，转为成PIL的格式的RGB图片
        if isinstance(img1, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            print(type(image))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(image)
        # 参数依次为 字体、字体大小、编码
        font_style = ImageFont.truetype("font/simsun.ttc", text_size, encoding='utf-8')
        # 参数依次为位置、文本、颜色、字体
        draw.text((left, top), text, text_color, font=font_style)

        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


    image = image_add_text(image, "哈儿", 225, 50, (255, 0, 0), 40)

    cv2.imshow(winname='image', mat=image)
