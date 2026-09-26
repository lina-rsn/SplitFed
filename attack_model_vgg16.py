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
class encoder_vgg16(nn.Module):
    def __init__(self):
            super(encoder_vgg16, self).__init__()
            # here load the pretrained on imagenet vgg16
            # cut the model on the cut layer
            vgg_model = models.vgg16(pretrained = True)
            features = list(vgg_model.features.children())
            self.model = nn.Sequential(*features[:5]) # includes Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d 

            for p in self.model.parameters(): # trying out the attack with freezed weights
                p.requires_grad_(False)
    
    def forward(self, x):
        return self.model(x)     


class decoder_vgg16(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.ConvTranspose2d(
            64, 64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.layer2 = nn.Conv2d(
            64, 64,
            kernel_size=3,
            padding=1
        )

        self.layer3 = nn.Conv2d(
            64, 32,
            kernel_size=3,
            padding=1
        )

        self.output = nn.Conv2d(
            32, 3,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 16 -> 32
        x = self.relu(self.layer2(x))  # 32 -> 32
        x = self.relu(self.layer3(x))  # 32 -> 32
        x = self.output(x)             # 32 -> 32

        return x

    
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder_vgg16()
        self.decoder = decoder_vgg16()

    def forward(self, x):
        z = self.encoder(x)
        # x_reconstructed = self.decoder(z)
        x_reconstructed = torch.sigmoid(self.decoder(z)) # for better colour reconstruction

        return x_reconstructed

###############################################################################################
# training
###############################################################################################
def to_img(x, dim):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 3, dim, dim)
        return x


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

_MEAN_T = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD_T  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

def denormalize_cifar10(x):
    return (x * _STD_T.to(x.device) + _MEAN_T.to(x.device)).clamp(0, 1)

def detransform_cifar10(x):
    x_de = denormalize_cifar10(x)
    x_de = transforms.Resize(
        (32, 32),
        interpolation=transforms.InterpolationMode.BILINEAR
    )(x_de)
    return x_de


def train_autoencoder(dataloaders, logger, device, wd, constant):
    model = autoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-5)

    for epoch in range(constant.attacker_epochs):
        model.train()
        for data in dataloaders['train']:
            img, _ = data
            img = img.to(device)

            target = denormalize_cifar10(img) # training in denormalized space for better colour reconstr.
            output = model(img)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, constant.attacker_epochs, loss.item()))
            #picbefore = to_img(img.data, img.data.shape[2])
            #picafter = to_img(output.data, img.data.shape[2])

            save_image(detransform_cifar10(img.data), RECON_DIR+ '/attacker_'+constant.INTERMEDIATE_DIR+'image_{}_before.png'.format(epoch))
            save_image(output, RECON_DIR+'/attacker_'+constant.INTERMEDIATE_DIR+'image_{}_after.png'.format(epoch))
    return model

def save_image_tensor(tensor, filename): #never used
    
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
    attacker_epochs = 40          # epochs to train the attacker's own autoencoder on CIFAR100
    INTERMEDIATE_DIR =  "intermediate_vgg16"
    RECON_DIR = "reconstruction_vgg16"
    ATTACK_DATA_DIR = "attack_vgg16"
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
 
INTERMEDIATE_DIR = os.path.join(wd,  constant.INTERMEDIATE_DIR)
ATTACK_DATA_DIR = os.path.join(wd, constant.ATTACK_DATA_DIR)
LABEL_DIR = os.path.join(wd, "labels_vgg16")
RECON_DIR = os.path.join(wd, constant.RECON_DIR)
IMAGE_DIR = os.path.join(wd, "images_vgg16") 

os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
Path(os.path.join(wd, constant.RECON_DIR)).mkdir(parents=True, exist_ok=True)  # used by train_autoencoder()
#Path(RECON_DIR).mkdir(parents=True, exist_ok=True)                          # used by the attack step below
 
# Load CIFAR 100 

'''
auxilary_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
 
cifar100_train = datasets.CIFAR100(root='data', train=True, download=True, transform=auxilary_transforms)
cifar100_val = datasets.CIFAR100(root='data', train=False, download=True, transform=auxilary_transforms)
'''

# Load CIFAR 10 

auxilary_transforms = transforms.Compose([
                        transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = CIFAR10_MEAN, std = CIFAR10_STD)
                        ])

dataset_train = datasets.CIFAR10(root='data', train=True, download=True, transform=auxilary_transforms)
dataset_test = datasets.CIFAR10(root='data', train=False, download=True, transform=auxilary_transforms)

 
trainloader = DataLoader(dataset_train, batch_size=constant.BATCH_SIZE, shuffle=True)
valloader = DataLoader(dataset_test, batch_size=constant.BATCH_SIZE, shuffle=False)
 
dataloaders = {'train': trainloader, 'val': valloader}
 
#print(f"[Attacker] Training reconstruction autoencoder on CIFAR100 for {constant.attacker_epochs} epochs")
print(f"[Attacker] Training reconstruction autoencoder on CIFAR10 for {constant.attacker_epochs} epochs")

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

 

 
 
reconstruction_count = 0
mse_sum, psnr_sum, scored_batches = 0.0, 0.0, 0
per_batch_results = []

with torch.no_grad():
    for filename in available_files:
        epoch_str, client_str, batch_str = filename.replace(".pt", "").split("_")
 
        intermediate = torch.load(os.path.join(INTERMEDIATE_DIR, filename), map_location=device)
        intermediate = intermediate.to(device)
 
        reconstructed = model.decoder(intermediate)
        #pic = denormalize_cifar10(reconstructed)
        pic = detransform_cifar10(reconstructed.data)
 
        out_name = f"epoch{epoch_str}_client{client_str}_batch{batch_str}.png"
        
        save_image(pic, os.path.join(ATTACK_DATA_DIR, out_name))
        reconstruction_count += 1

        image_path = os.path.join(IMAGE_DIR, filename)
        if os.path.isfile(image_path):
            original = torch.load(image_path, map_location=device)
            original_01 = detransform_cifar10(original)
            out_name = f"epoch{epoch_str}_client{client_str}_batch{batch_str}_org.png"
            save_image(original_01, os.path.join(ATTACK_DATA_DIR, out_name))
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