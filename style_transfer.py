import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Show image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Load images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

# VGG model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Layers
content_layers = ['21']
style_layers = ['0', '5', '10', '19', '28']

# Feature extraction
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers:
            features['content'] = x
        if name in style_layers:
            features[name] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Extract features
content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)

# Style gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Target image (start from content)
target = content_img.clone().requires_grad_(True).to(device)

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Weights
content_weight = 1e4
style_weight = 1e2

# Training loop
steps = 300

for i in range(steps):
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['content'] - content_features['content'])**2)

    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]

        _, d, h, w = target_feature.shape
        layer_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_loss / (d * h * w)

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}, Loss: {total_loss.item()}")

# Show final image
imshow(target, title="Styled Image")
