import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
## Add more imports if required

####################################################################################################################
# Define your model and transform and all necessary helper functions here                                          #
# They will be imported to the exp_recognition.py file                                                             #
####################################################################################################################

# Definition of classes as dictionary
classes = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}

# Expression CNN - EXACT same architecture as trained in Colab (notebook 4b)
# Conv blocks + global average pooling + 1 Linear layer.
# widths/drops come from the 'config' saved inside expression_model.t7 -> always matches the trained model
# Supports BOTH saved model types (chosen by the 'config' inside expression_model.t7):
#   v2  (own CNN)        : config = {widths, drops, head_drop}  -> layers features / pool / classifier
#   v3  (Colab ResNet18) : config = {backbone, head_drop}       -> layers net / head  (same names as Colab v3)
class facExpRec(torch.nn.Module):
    def __init__(self, num_classes=7, widths=(32, 64, 128, 256), drops=(0.1, 0.2, 0.3, 0.3), head_drop=0.5,
                 backbone=None, pretrained=False):
        super(facExpRec, self).__init__()
        self.backbone_name = backbone
        if backbone is not None:          # ---- v3: pretrained backbone, rebuilt WITHOUT ImageNet weights
            if backbone == 'resnet18':
                net = models.resnet18(weights=None)
                feat = net.fc.in_features
                net.fc = nn.Identity()
            elif backbone == 'mobilenet_v2':
                net = models.mobilenet_v2(weights=None)
                feat = net.classifier[-1].in_features
                net.classifier = nn.Identity()
            else:
                raise ValueError("unknown backbone: " + str(backbone))
            self.net = net
            self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(feat, num_classes))
            return
        def block(c_in, c_out, drop):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(drop))
        chans = (1,) + tuple(widths)
        self.features = nn.Sequential(*[block(chans[i], chans[i + 1], drops[i]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(head_drop), nn.Linear(widths[-1], num_classes))

    def forward(self, x):
        if self.backbone_name is not None:
            return self.head(self.net(x))
        return self.classifier(self.pool(self.features(x)))

# Sample Helper function
def rgb2gray(image):
    return image.convert('L')
    
# Sample Transformation function
#YOUR CODE HERE for changing the Transformation values.
# NOTE: exp_recognition.py uses make_transform(img_size) with the size saved in expression_model.t7 (v2 = 64).
# trnscm is kept only as the template default/fallback.
trnscm = transforms.Compose([rgb2gray, transforms.Resize((64,64)), transforms.ToTensor()])

# Same transform for any image size saved in the model file (e.g. 48 or 64)
def make_transform(img_size=48):
    return transforms.Compose([rgb2gray, transforms.Resize((img_size, img_size)), transforms.ToTensor()])


# ---------------- v3 (Colab ResNet18) transform: MUST match eval_tf in Hackathon4b_Expression_Recognition_v3 ----------------
def to_gray3(im):
    # detected_face() returns a mode-'L' image; the backbone needs 3 identical channels
    return im.convert('L').convert('RGB')

class SquarePad:
    """Letterbox to a square with the image's own mean colour (keeps face proportions)."""
    def __call__(self, im):
        w, h = im.size
        if w == h:
            return im
        s = max(w, h)
        fill = tuple(int(v) for v in ImageStat.Stat(im).mean)
        if im.mode == 'L':
            fill = fill[0]
        canvas = Image.new(im.mode, (s, s), fill)
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        return canvas

def make_backbone_transform(img_size=96, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([to_gray3, SquarePad(), transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(), transforms.Normalize(list(mean), list(std))])


# ---------------- OPTIONAL v3: transfer learning (NOT deployed now; v2 CNN is the baseline) ----------------
# IDENTICAL to build_tl_model in notebook Hackathon4b_Expression_Experiments. Only used if expression_model.t7 contains 'arch'. On the server pretrained=False (no ImageNet download);
# our fine-tuned weights are loaded afterwards from expression_model.t7
def build_tl_model(arch='efficientnet_b0', num_classes=7, pretrained=False):
    if arch == 'efficientnet_b0':
        m = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == 'mobilenet_v3_large':
        m = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m

# IDENTICAL to tl_eval_tf in notebook Hackathon4b_Expression_Experiments: grayscale -> 3 channels -> resize -> tensor -> ImageNet normalisation
def make_tl_transform(img_size=112, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.Grayscale(num_output_channels=3),
                               transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(list(mean), list(std))])