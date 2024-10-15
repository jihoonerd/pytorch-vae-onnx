import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os

# Set random seed for reproducibility
torch.manual_seed(42)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
batch_size = 128
latent_dim = 20
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # mu
        self.fc22 = nn.Linear(400, latent_dim)  # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Initialize the VAE and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


# Run the training
for epoch in range(1, epochs + 1):
    train(epoch)


# Generate new MNIST digits
def generate_digits(num_samples=10):
    with torch.no_grad():
        sample = torch.randn(num_samples, latent_dim).to(device)
        generated = model.decode(sample).cpu()
        return generated.view(-1, 1, 28, 28)


# Generate and save digits
def save_generated_digits(num_samples=10, output_dir='generated_digits'):
    generated_digits = generate_digits(num_samples)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    for i, digit in enumerate(generated_digits):
        utils.save_image(digit, os.path.join(output_dir, f'generated_digit_{i}.png'))

    # Save a grid of images
    utils.save_image(generated_digits, os.path.join(output_dir, 'generated_digits_grid.png'), nrow=5)

    print(f"Generated digits saved in {output_dir}")


# Generate and save digits
save_generated_digits(num_samples=25)


# Export model to ONNX
def export_to_onnx(model, output_path='vae_mnist.onnx'):
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("ONNX or ONNX Runtime is not installed. Please install them using:")
        print("pip install onnx onnxruntime")
        print("Then run this function again.")
        return

    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    try:
        # Export the encoder
        torch.onnx.export(model,
                          dummy_input,
                          output_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['reconstructed', 'mu', 'logvar'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'reconstructed': {0: 'batch_size'},
                                        'mu': {0: 'batch_size'},
                                        'logvar': {0: 'batch_size'}})

        print(f"Model exported to ONNX format at {output_path}")

        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully!")

        # Test with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        print("ONNX Runtime inference test passed successfully!")

    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        print("Please ensure you have the latest versions of PyTorch and ONNX installed.")


# Export the model to ONNX
export_to_onnx(model)