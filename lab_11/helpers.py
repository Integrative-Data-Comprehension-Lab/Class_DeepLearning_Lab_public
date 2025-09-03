import matplotlib.pyplot as plt
import torch

def visualize_few_samples(dataset, cols=8, rows=5):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR10 class names

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 1, rows * 1)) 
    axes = axes.flatten()

    for i in range(cols * rows):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        img = img.permute(1, 2, 0)  # CHW to HWC
        img = img.numpy()  # Convert to numpy array
        img = (img * 0.5 + 0.5)  # Unnormalize to [0,1] for display
        axes[i].imshow(img)
        axes[i].set_title(class_names[label])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()