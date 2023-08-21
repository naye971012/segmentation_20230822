from losses import *
from tqdm import tqdm
from utills import compute_miou

def train(config, model, logger, train_dataloader, vali_dataloader):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    
    model = model.to(device)
    for epoch in range(config['EPOCH']):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        iou_list = torch.zeros(config['DATASET']['NUM_CLASSES']+1) #각 class별 IOU
        for i, (images, masks) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            iou_list += compute_miou(outputs[1],masks)
            epoch_loss += loss.item()
            if(i%50==49):
                logger.add_scalar('train loss step', epoch_loss/i , epoch * len(train_dataloader) + i )
                logger.add_scalar('train mIOU step', iou_list[config['DATASET']['NUM_CLASSES']]/i , epoch * len(train_dataloader) + i )
                for j in range(config['DATASET']['NUM_CLASSES']):
                    logger.add_scalar(f'train IOU class {j} step', iou_list[j]/i , epoch * len(train_dataloader) + i )
        
        scheduler.step()
        
        print("lr: ", optimizer.param_groups[0]['lr'])
        logger.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        
        print("loss: ", epoch_loss/len(train_dataloader) , epoch)
        logger.add_scalar('train loss total', epoch_loss/len(train_dataloader) , epoch )
        logger.add_scalar('train mIOU total', iou_list[config['DATASET']['NUM_CLASSES']]/len(train_dataloader) , epoch )
        for j in range(config['DATASET']['NUM_CLASSES']):
            logger.add_scalar(f'train IOU class {j} total', iou_list[j]/len(train_dataloader) , epoch )

def vali():
    pass

def test():
    pass