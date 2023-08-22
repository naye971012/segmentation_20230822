import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 클래스별 색상 매핑 (예: 26개 클래스에 대한 색상)
class_colors = [
    [0, 0, 255],    # 클래스 0의 색상 (파랑)
    [255, 0, 0],    # 클래스 1의 색상 (빨강)
    [0, 255, 0],    # 클래스 2의 색상 (녹색)
    [255, 255, 0],  # 클래스 3의 색상 (노랑)
    [0, 255, 255],  # 클래스 4의 색상 (하늘색)
    [255, 0, 255],  # 클래스 5의 색상 (분홍색)
    [128, 0, 0],    # 클래스 6의 색상 (짙은 빨강)
    [0, 128, 0],    # 클래스 7의 색상 (짙은 녹색)
    [0, 0, 128],    # 클래스 8의 색상 (짙은 파랑)
    [128, 128, 0],  # 클래스 9의 색상 (올리브)
    [128, 0, 128],  # 클래스 10의 색상 (자주색)
    [0, 128, 128],  # 클래스 11의 색상 (청록색)
    [192, 192, 192],  # 클래스 12의 색상 (은색)
    [128, 128, 128],  # 클래스 13의 색상 (회색)
    [0, 0, 0],      # 클래스 14의 색상 (검정)
    [255, 255, 255],  # 클래스 15의 색상 (흰색)
    [165, 42, 42],   # 클래스 16의 색상 (갈색)
    [255, 140, 0],   # 클래스 17의 색상 (주황)
    [255, 215, 0],   # 클래스 18의 색상 (금색)
    [218, 112, 214],  # 클래스 19의 색상 (보라)
    [0, 255, 0],     # 클래스 20의 색상 (라임색)
    [0, 0, 139],     # 클래스 21의 색상 (어두운 파랑)
    [0, 128, 0],     # 클래스 22의 색상 (녹색)
    [70, 130, 180],  # 클래스 23의 색상 (스카이 블루)
    [102, 205, 170],  # 클래스 24의 색상 (민트 크림)
    [176, 224, 230],  # 클래스 25의 색상 (파우더 블루)
]

def seed_everything(config):
    random.seed(0)
    torch.manual_seed(0)    
    cudnn.benchmark = config['CUDNN']['BENCHMARK']
    cudnn.deterministic = config['CUDNN']['DETERMINISTIC']
    cudnn.enabled = config['CUDNN']['ENABLED']
    print("seed set to 0")
    

def compute_miou(pred, true_labels, num_classes=26, idx=-1, is_validation=False):
    """
    전체 IOU list는 26번째에, 각 index마다 각각 class의 IOU

    Args:
        pred_labels (_type_): _description_
        true_labels (_type_): _description_
        num_classes (int, optional): _description_. Defaults to 26.

    Returns:
        _type_: _description_
    """
    ph, pw = pred.size(2), pred.size(3)
    h, w = true_labels.size(2), true_labels.size(3)
    if ph != h or pw != w:
        pred = F.interpolate(input=pred, size=( #여기서 크기 조정하네
                h, w), mode='bilinear', align_corners=True)
    
    
    pred_labels = (pred > 0.5).to(torch.int)
    true_labels = true_labels.to(torch.int)
    
    if(idx%50==0):
        draw_image(pred_labels,true_labels, is_validation, idx)
    
    mean_iou_list =0.0
    epsilon = 1e-6  # 분모가 0이 되는 것을 방지하기 위한 작은 값

    iou_list = torch.zeros(num_classes+1) #각 class별 IOU
    perfect_list = torch.zeros(num_classes+1) #각 class별 IOU
    
    for batch in range(pred_labels.shape[0]):
        iou_per_batch = 0.0
        for c in range(num_classes):  
            true_mask = true_labels[batch][c]
            pred_mask = pred_labels[batch][c]
            
            intersection = torch.logical_and(true_mask, pred_mask).sum()
            union = torch.logical_or(true_mask, pred_mask).sum()
            
            iou_per_class = (intersection + epsilon) / (union + epsilon)
            
            if(true_mask.sum()==0 and pred_mask.sum()==0): #없는 상황에서 예측 잘 하면
                perfect_list[num_classes]+=1
                perfect_list[c]+=1 # 이들은 mIOU 계산에서 제외 (추후 나눌 때 이들 개수만큼 빼서 나눔 제외)
            else:
                iou_per_batch += iou_per_class.cpu()
                iou_list[c] += iou_per_class.cpu()
                
                
        iou_list[num_classes] += (iou_per_batch/(num_classes - perfect_list[num_classes]))  #마지막 index (num_class) = 평균 iou
    perfect_list[num_classes] = 0 #바로 윗 줄에서 계산 완료했으므로 다시 사용 안되도록 0으로 바꿈

    return iou_list , perfect_list


    
def draw_image(pred,mask, is_validation, idx ):
    for i, mask_tensor in enumerate([mask,pred]):

        # 클래스별로 이미지 시각화 및 저장
        class_labels = (mask_tensor[0]).cpu().numpy()

        # 클래스별로 픽셀 수 계산, label일 경우에만 실행하여 같은 색으로
        if i==0:
            class_pixel_counts = [np.sum(class_labels[c] == 1) for c in range(26)]
        
        # 클래스별로 이미지 생성
        colored_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for color, c in enumerate(np.argsort(class_pixel_counts)[::-1]):  # 가장 많이 등장한 순서대로 순회
            class_mask = (class_labels[c] == 1)
            class_color = class_colors[color]
            colored_image[class_mask] = class_color

        # 이미지 시각화
        plt.imshow(colored_image)
        plt.axis('off')
        
        if is_validation:
            prefix = "vali_"
        else:
            prefix = "train_"
        if i==0:
            name = f'{prefix}label_visualization_{idx}'
            plt.savefig(f'{name}.png')  # 이미지 저장
            save_each_class(class_labels,name)
        else:
            name = f'{prefix}pred_visualization_{idx}'
            plt.savefig(f'{name}.png')  # 이미지 저장
            save_each_class(class_labels,name)

def save_each_class(selected_images,name):

    # subplot 생성 및 데이터 채우기
    fig, axes = plt.subplots(6, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if(i==26):
            break
        ax.imshow(selected_images[i], cmap='gray')  # 흑백 이미지로 표시
        ax.axis('off')  # 축 제거

    # 그래프 저장
    plt.tight_layout()
    plt.savefig(f'each_{name}.png')

def calculate_weight(train_dataset):
    """
    data 5000개 분포 확인하여 class별 weight설정

    Args:
        train_dataset (_type_): _description_
    """
    class_pixel_counts = torch.zeros(26).to('cuda')
    for __ in tqdm(range(100)):
        i = random.randint(0,len(train_dataset)-1)
        _ , label = train_dataset[i]
        label = torch.tensor(label).to('cuda')
        class_pixel_counts += torch.tensor([torch.sum(label[c]) for c in range(26)]).to('cuda') #각 class별 개수
        #10000 100 500 6000
    class_pixel_counts = 1 - class_pixel_counts/torch.sum(class_pixel_counts)
    
    print(f"class weights : {class_pixel_counts}")
    
    return class_pixel_counts


import numpy as np
import matplotlib.pyplot as plt

# 랜덤한 [26, 1024, 1024] 크기의 0과 1로 이루어진 텐서 생성 (이 부분은 당신의 데이터로 대체해주세요)
tensor = np.random.randint(2, size=(26, 1024, 1024))

