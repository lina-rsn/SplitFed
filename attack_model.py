############################################################
# attack scripts
# no defence version
# adversary's knowledge:
# 1) clients model architecture (no weights)
# 2) access to very similar dataset (CIFAR10 - CIFAR100)
###############################################################################################
# attack code is based on: https://github.com/HEECMA-BU/FSL/tree/main
###############################################################################################

import torch
from torch import nn
import torchvision
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path


target_h = 32
target_w = 32

###############################################################################################
# encoder - decoder architecture
###############################################################################################

class autoencoder_resnet18(nn.Module):
    def __init__(self):
            super(autoencoder_resnet18, self).__init__()
            self.layer1 = nn.Sequential (
                    nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU (inplace = True),
                    nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
                )
            self.layer2 = nn.Sequential  (
                    nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU (inplace = True),
                    nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(64),
                )
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
    
    
    def forward(self, x):
            resudial1 = F.relu(self.layer1(x))
            out1 = self.layer2(resudial1)
            out1 = out1 + resudial1 
            resudial2 = F.relu(out1)
            return resudial2


class decoder_resnet18(nn.Module):
    def __init__(self):
        super(decoder_resnet18, self).__init__()

        # Mirrors encoder's layer2 (the residual block) — this becomes
        # the *first* thing the decoder does, since decoder runs in reverse
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # Upsample x2 (undoes encoder's maxpool stride=2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                                padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Upsample x2 again (undoes encoder's first conv stride=2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2,
                                padding=3, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Project back to 3 channels (RGB)
        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Mirror the encoder's residual block
        residual = x
        out = self.layer1(x)
        out = out + residual        # "adding the input here again"
        out = F.relu(out)

        # Undo the encoder's two stride-2 downsampling steps
        out = self.up1(out)         # /4 -> /2 resolution
        out = self.up2(out)         # /2 -> full resolution

        out = torch.tanh(self.output(out))  # or leave unbounded / use tanh
        return out

'''
def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x

    class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

    
'''

'''class decoder_resnet18(nn.Module):
    def __init__(self):
        super(decoder_resnet18, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                64, 64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True) 
        )

        # 64 feature maps to
        self.output = nn.Conv2d(
            64, 3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Inverse residual block
        residual = x

        out = self.layer2(x)
        out = out + residual
        out = F.relu(out)

        # 1/2 resolution
        out = self.up1(out)

        # full resolution
        out = self.up2(out)

        out = self.output(out)

        return out '''

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = autoencoder_resnet18()
        self.decoder = decoder_resnet18()

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed

###############################################################################################
# training
###############################################################################################
def to_img(x, dim):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 3, dim, dim)
        return x

def train_autoencoder(dataloaders, logger, device, wd, constant):
    model = autoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-5)

    for epoch in range(constant.attacker_epochs):
        for data in dataloaders['train']:
            img, _ = data
            img = img.to(device)

            output = model.forward(img)
            loss = criterion(output,img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, constant.attacker_epochs, loss.item()))
            picbefore = to_img(img.data, img.data.shape[2])
            picafter = to_img(output.data, img.data.shape[2])

            save_image(picbefore, wd+'/reconstruction/attacker_'+constant.INTERMEDIATE_DATA_DIR+'image_{}_before.png'.format(epoch))
            save_image(picafter, wd+'/reconstruction/attacker_'+constant.INTERMEDIATE_DATA_DIR+'image_{}_after.png'.format(epoch))
    return model

def save_image_tensor(tensor, filename):
    
    tensor = tensor.detach().cpu()
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor * 255).byte()
    image = Image.fromarray(tensor.numpy())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)

def compute_mse_psnr(reconstructed_01, original_01):
    """MSE and PSNR between two batches of images already scaled to [0, 1].
    PSNR uses a peak value of 1.0 since inputs are in [0, 1]."""
    mse = F.mse_loss(reconstructed_01, original_01).item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
    return mse, psnr

################################################
# attack implementation
################################################

class Config:
    attacker_epochs = 30          # epochs to train the attacker's own autoencoder on CIFAR100
    INTERMEDIATE_DATA_DIR = "Train"
    ATTACK_DATA_DIR = "Attack"
    SAVE_EVERY_N_EPOCHS = 50
    BATCH_SIZE = 256
    SEED = 1234
 
constant = Config()
wd = "."   
 
torch.manual_seed(constant.SEED)
 
if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Is the current version of PyTorch built with MPS activated?",torch.backends.mps.is_built())
else:
    device = 'cpu'

print(f"[Attacker] Using device: {device}")
 
INTERMEDIATE_DIR = os.path.join(wd, "intermediate", constant.INTERMEDIATE_DATA_DIR)
ATTACK_DATA_DIR = os.path.join(wd, "attacker_" + constant.ATTACK_DATA_DIR)
LABEL_DIR = os.path.join(wd, "labels", constant.INTERMEDIATE_DATA_DIR)
RECON_DIR = os.path.join(wd, "reconstruction", "attacker_" + constant.INTERMEDIATE_DATA_DIR)
IMAGE_DIR = os.path.join(wd, "images", constant.INTERMEDIATE_DATA_DIR) 

os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
Path(os.path.join(wd, "reconstruction")).mkdir(parents=True, exist_ok=True)  # used by train_autoencoder()
Path(RECON_DIR).mkdir(parents=True, exist_ok=True)                          # used by the attack step below
 
# Load CIFAR 100 

auxilary_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
 
cifar100_train = datasets.CIFAR100(root='data', train=True, download=True, transform=auxilary_transforms)
cifar100_val = datasets.CIFAR100(root='data', train=False, download=True, transform=auxilary_transforms)
 
trainloader = DataLoader(cifar100_train, batch_size=constant.BATCH_SIZE, shuffle=True)
valloader = DataLoader(cifar100_val, batch_size=constant.BATCH_SIZE, shuffle=False)
 
dataloaders = {'train': trainloader, 'val': valloader}
 
print(f"[Attacker] Training reconstruction autoencoder on CIFAR100 for {constant.attacker_epochs} epochs")
model = train_autoencoder(dataloaders, logger=None, device=device, wd=wd, constant=constant)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
 
# attack

if not os.path.isdir(INTERMEDIATE_DIR):
    raise FileNotFoundError(
        f"No intermediate data found at {INTERMEDIATE_DIR}. "
    )
 
available_files = sorted(f for f in os.listdir(INTERMEDIATE_DIR) if f.endswith(".pt"))
available_epochs = sorted(set(int(f.split("_")[0]) for f in available_files))
print(f"[Attacker] Found saved intermediate data for epochs: {available_epochs}")
#The learner normalized images with CIFAR10's own mean/std before feeding them to the
# client model. To compare fairly against the reconstruction (which to_img() puts in [0, 1]),
# ground-truth images need to be denormalized back to [0, 1] first.
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
 
def denormalize_cifar10(x):
    return (x * CIFAR10_STD.to(x.device) + CIFAR10_MEAN.to(x.device)).clamp(0, 1)
 
 
reconstruction_count = 0
mse_sum, psnr_sum, scored_batches = 0.0, 0.0, 0
per_batch_results = []

with torch.no_grad():
    for filename in available_files:
        epoch_str, client_str, batch_str = filename.replace(".pt", "").split("_")
 
        intermediate = torch.load(os.path.join(INTERMEDIATE_DIR, filename), map_location=device)
        intermediate = intermediate.to(device)
 
        reconstructed = model.decoder(intermediate)
        pic = to_img(reconstructed.data, reconstructed.data.shape[2])
 
        out_name = f"epoch{epoch_str}_client{client_str}_batch{batch_str}.png"
        
        save_image(pic, os.path.join(ATTACK_DATA_DIR, out_name))
        reconstruction_count += 1

        image_path = os.path.join(IMAGE_DIR, filename)
        if os.path.isfile(image_path):
            original = torch.load(image_path, map_location=device)
            original_01 = denormalize_cifar10(original)
 
            batch_mse, batch_psnr = compute_mse_psnr(pic, original_01)
            mse_sum += batch_mse
            psnr_sum += batch_psnr
            scored_batches += 1
            per_batch_results.append((filename, batch_mse, batch_psnr))
 
            print(f"[Attacker] {filename}: MSE={batch_mse:.4f}, PSNR={batch_psnr:.2f} dB")
 
print(f"[Attacker] Reconstructed {reconstruction_count} batches of images from captured intermediate data.")
print(f"[Attacker] Reconstructions saved to: {ATTACK_DATA_DIR}")
 
if scored_batches > 0:
    avg_mse = mse_sum / scored_batches
    avg_psnr = psnr_sum / scored_batches
    print(f"[Attacker] FINAL reconstruction quality against real victim images "
          f"({scored_batches} batches): avg MSE={avg_mse:.4f}, avg PSNR={avg_psnr:.2f} dB")
    

###############################################################################################
# test - generated
###############################################################################################

'''
if torch.cuda.is_available():
     device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Is the current version of PyTorch built with MPS activated?",torch.backends.mps.is_built())
else:
    device = 'cpu'

print("Using device:", device)


# ============================================================
# 5. CIFAR-10 DATASET
# ============================================================

transform = transforms.ToTensor()


trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = autoencoder().to(device)

print(model)


# ============================================================
# 7. CHECK SHAPES
# ============================================================

dummy = torch.randn(
    4, 3, 32, 32
).to(device)


with torch.no_grad():

    latent = model.encoder(dummy)

    reconstructed = model.decoder(latent)


print()
print("Input shape:        ", dummy.shape)
print("Latent shape:       ", latent.shape)
print("Reconstruction:     ", reconstructed.shape)
print()


# Expected:
#
# Input shape:        [4, 3, 32, 32]
# Latent shape:       [4, 64, 8, 8]
# Reconstruction:     [4, 3, 32, 32]


# ============================================================
# 8. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    for images, _ in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(images)

        # Reconstruction loss
        loss = criterion(
            reconstructed,
            images
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    epoch_loss = (
        running_loss
        / len(dataloader.dataset)
    )

    return epoch_loss


# ============================================================
# 10. TEST / VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(
                device,
                non_blocking=True
            )

            reconstructed = model(images)

            loss = criterion(
                reconstructed,
                images
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

    epoch_loss = (
        running_loss
        / len(dataloader.dataset)
    )

    return epoch_loss


# ============================================================
# 11. TRAIN
# ============================================================

num_epochs = 40

train_losses = []
test_losses = []


for epoch in range(num_epochs):

    train_loss = train_one_epoch(
        model,
        trainloader,
        optimizer,
        criterion,
        device
    )


    test_loss = evaluate(
        model,
        testloader,
        criterion,
        device
    )


    train_losses.append(train_loss)
    test_losses.append(test_loss)


    print(
        f"Epoch [{epoch+1:02d}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Test Loss: {test_loss:.6f}"
    )


# ============================================================
# 12. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "cifar10_autoencoder.pth"
)

print("\nModel saved.")





# ============================================================
# 14. RECONSTRUCTION VISUALIZATION
# ============================================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


model.eval()


images, labels = next(
    iter(testloader)
)


images = images.to(device)


with torch.no_grad():

    reconstructions = model(images)


# Show first 8 images

fig, axes = plt.subplots(
    2,
    8,
    figsize=(16, 4)
)


for i in range(8):

    # -----------------------------
    # Original
    # -----------------------------

    original = images[i].cpu()

    axes[0, i].imshow(
        original.permute(1, 2, 0)
    )

    axes[0, i].set_title(
        class_names[labels[i]]
    )

    axes[0, i].axis("off")


    # -----------------------------
    # Reconstruction
    # -----------------------------

    reconstruction = (
        reconstructions[i]
        .cpu()
        .clamp(0, 1)
    )

    axes[1, i].imshow(
        reconstruction.permute(1, 2, 0)
    )

    axes[1, i].set_title(
        "Reconstruction"
    )

    axes[1, i].axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 15. CALCULATE PSNR
# ============================================================

mse = test_losses[-1]

psnr = 10 * math.log10(
    1.0 / mse
)

print(
    f"Final test MSE:  {mse:.6f}"
)

print(
    f"Final test PSNR: {psnr:.2f} dB"
)'''