from .resnet50_fpn_model import *
import torch
from .faster_rcnn_framework import *
from .transforms import *

def create_model(num_classes,load_pretrain_weights=False):
    backbone=resnet50_fpn_backbone(pretrain_path='',norm_layer=torch.nn.BatchNorm2d,layer_to_train=3)
    model=FasterRCNN(backbone=backbone,num_classes=91)

    if load_pretrain_weights:
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main(args):
    device=torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    print(("using {} device training".format(device.type)))

    data_transform={
        'train':transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(0.5)]),
        'val':transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    if args.aspect_ratio_group_factor>=0:
        train_sampler=torch.utils.data.RandomSampler(train_dataset)  #将所有的数据顺序打乱，
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size=args.batch_size
    nw=min([os.cpu_count(),batch_size if batch_size>1 else 0,8])
    if train_sampler:
        train_data_loader=torch.utils.data.Dataloader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_works=nw,
            collate_fn=train_dataset.collate_fn

        )
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    model=create_model(num_classes=args.num_classes + 1)
    model.to(device)

    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    scaler=torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.33)

    if args.resume!="":
        checkpoint=torch.load(args.resume,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch=checkpoint['epoch']+1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])


    train_loss=[]
    learning_rate=[]
    val_map=[]

    for epoch in range(args.start_epoch,args.epochs):
        mean_loss,lr=utils.train_one_epoch(model,optimizer,train_data_loader,device=device,epoch=epoch,print_freq=50,warmup=True,scaler=scaler)








