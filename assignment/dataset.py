"""
Loads CIFAR-10 and returns flattened, normalized image vectors.

We flatten to (N, 3072) because PCA operates on vectors, and using the
same flattened representation for the autoencoder keeps the comparison
fair — both methods compress the exact same input representation.
"""
import torch
import torchvision

def load_cifar_flat(data_dir="./data", train=True):
    """
    Loads CIFAR-10 and returns flattened, normalized image vectors.

    Args:
        data_dir (str): Directory to download/load the CIFAR-10 dataset.

    Returns:
        torch.Tensor: Flattened and normalized image vectors of shape (N, 3072).
        torch.Tensor: Corresponding labels of shape (N,).
        X: (N, 3072) float tensor, pixel values normalized to [0, 1].
    """
    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
    # dataset.data is a numpy array of shape (N, 32, 32, 3), unit8 values in [0, 255]
    imgs = torch.from_numpy(dataset.data).float() / 255.0 #(N, 32, 32, 3) float tensor in [0, 1]
    X = imgs.reshape(imgs.shape[0], -1) #(N, 3072) float tensor in [0, 1]
    return X

def unflatten(x, img_shape=(32, 32, 3)):
    """
    Unflattens a flattened image vector back to its original shape.

    Args:
        x (torch.Tensor): Flattened image vector of shape (N, 3072).
        img_shape (tuple): Original shape of the image (height, width, channels).

    Returns:
        torch.Tensor: Unflattened image tensor of shape (N, height, width, channels).
        (N, 3072) -> (N, 32, 32, 3) for visualization
    """
    return x.reshape(-1, *img_shape).clamp(0, 1) # Ensure pixel values are in [0, 1]

if __name__ == "__main__":
    # Example usage
    X = load_cifar_flat()
    print("Flattened CIFAR-10 shape:", X.shape)  # Should print (N, 3072)
    unflattened_X = unflatten(X)
    print("Unflattened CIFAR-10 shape:", unflattened_X.shape)  # Should print (N, 32, 32, 3)