"""
Pure-PyTorch PCA for image compression, using torch.pca_lowrank
(no sklearn — this is an approximate but efficient and accurate
low-rank SVD, well suited to datasets the size of CIFAR-10).
"""
class TORCHPCA:
    def __init__(self, n_components: int):
        """_summary_

        Args:
            n_components (int): _description_
        """
        self.n_components = n_components
        self.mean_ = None 
        self.components_ = None # (n_features, n_components)
        
    def fit(self, X: torch.Tensor):
        """
        fits the PCA model to the data X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where n_samples is the number of samples and n_features is 
            the number of features.
        """
        self.mean_ = X.mean(dim=0, keepdim=True) # (1, n_features)
        X_centered = X - self.mean_ # (n_samples, n_features)
        # Compute the low-rank SVD using torch.pca_lowrank  
        _, _, V = torch.pca_lowrank(X_centered, q=self.n_components, niter=niter)
        self.components_ = V[:, :self.n_components] # (n_features, n_components)  