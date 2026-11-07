import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# Add more imports if required

# Sample Transformation function
# YOUR CODE HERE for changing the Transformation values.
trnscm = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])

## Siamese Network - EXACT same architecture as trained in Colab (Stage 1)
## Layer names (cnn1, fc1) must match the saved state_dict keys in siamese_model.t7
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),             # 100x100 -> 50x50

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),             # 50x50 -> 25x25
        )

        # fc1 input 8*25*25 = 5,000 (was 80,000) => model ~10MB instead of ~160MB
        self.fc1 = nn.Sequential(
            nn.Linear(8*25*25, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    # forward_once: embedding for ONE image (also used for Stage 3 classifier features)
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2=None):
        output1 = self.forward_once(input1)
        if input2 is None:
            return output1
        output2 = self.forward_once(input2)
        return output1, output2

# Alias so the template name 'Siamese' also works
Siamese = SiameseNetwork

##########################################################################################################
## Sample classification network (Specify if you are using a pytorch classifier during the training)    ##
## classifier = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear...)           ##
##########################################################################################################

# YOUR CODE HERE for pytorch classifier

# Definition of classes as dictionary
classes = ['person1','person2','person3','person4','person5','person6','person7']