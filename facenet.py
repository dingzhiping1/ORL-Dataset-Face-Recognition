import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

def img_initialization(img_input):
    # Get original size of image
    # Percentage of the original size
    width = int(160)
    height = int(160)
    dim = (width, height)
    # Resize/Scale the image
    img_output = cv2.resize(img_input, dim, interpolation=cv2.INTER_AREA)
    # The new size of the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_output

# 获得人脸特征向量
def load_known_faces(dstImgPath, mtcnn, resnet):
    aligned = []
    knownImg = cv2.imread(dstImgPath)  # 读取图片
    print(knownImg.shape)
    knownImg = img_initialization(knownImg)
    face = mtcnn(knownImg)  # 使用mtcnn检测人脸，返回【人脸数组】

    if face is not None:
        aligned.append(face[0])
    aligned = torch.stack(aligned)
    with torch.no_grad():
        known_faces_emb = resnet(aligned).detach().cpu()  # 使用resnet模型获取人脸对应的特征向量

    return known_faces_emb, knownImg


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()
    if (distance < threshold):
        isExistDst = True
        print("\n两张人脸的欧式距离为：%.2f,小于阈值%f" % (distance,threshold))
    return isExistDst


if __name__ == '__main__':
    # help(MTCNN)
    # help(InceptionResnetV1)
    # mtcnn模型加载【设置网络参数，进行人脸检测】
    mtcnn = MTCNN(min_face_size=160, thresholds=[0.2, 0.2, 0.8], keep_all=True)
    # InceptionResnetV1模型加载【用于获取人脸特征向量】
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    MatchThreshold = 0.8  # 人脸特征向量匹配阈值设置
    examples = []
    labels = []
    for i in range (5,36):
        known_faces_emb, _ = load_known_faces('ORL Faces Database/s'+str(i+5)+'/1.bmp', mtcnn, resnet)  # 已知人物图
        examples.append(known_faces_emb)
        labels.append(i+5)
    # bFaceThin.png  lyf2.jpg
    faces_emb, img = load_known_faces('ORL Faces Database/s33/2.bmp', mtcnn, resnet)
    print("正在比对")# 待检测人物图
    for i in range(31):
        isExistDst = match_faces(faces_emb, examples[i], MatchThreshold)  # 人脸匹配
        if isExistDst:
            boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)  # 返回人脸框，概率，5个人脸关键点
            print('由于欧氏距离小于匹配阈值，故匹配,图片为第%d个人'% (i+10))
            break
        if isExistDst == False and i == 30:
            print('图片不属于任何库中的人')
