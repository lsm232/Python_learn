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


