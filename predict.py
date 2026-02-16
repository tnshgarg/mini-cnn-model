import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN

def predict():
    # Check for MPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    net = SimpleCNN().to(device)
    net.load_state_dict(torch.load('./cifar_net.pth', map_location=device, weights_only=True)) # Load trained weights
    net.eval() # Set to evaluation mode

    # Load Test Data (to pick a random image)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get a batch of random images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Move images to device for prediction
    images_device = images.to(device)

    # Predict
    outputs = net(images_device)
    _, predicted = torch.max(outputs, 1)

    # Calculate probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # Show results for the first image in the batch
    idx = 0
    img = images[idx] / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    
    print(f"\n--- Prediction for Image {idx+1} ---")
    print(f"Actual Label:    {classes[labels[idx]]}")
    print(f"Predicted Label: {classes[predicted[idx]]}")
    print("Confidence Scores:")
    for i in range(10):
        print(f"  {classes[i]:<10}: {probs[idx][i].item()*100:.2f}%")

    # Display image (if running locally with display)
    try:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f"Actual: {classes[labels[idx]]} | Pred: {classes[predicted[idx]]}")
        plt.show()
    except Exception as e:
        print(f"Check the plot window or saved image. (Display error: {e})")

if __name__ == "__main__":
    predict()
