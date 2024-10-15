import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
batch_size = 128
latent_dim = 20
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
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
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
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
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )


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
def save_generated_digits(num_samples=10, output_dir="generated_digits"):
    generated_digits = generate_digits(num_samples)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    for i, digit in enumerate(generated_digits):
        utils.save_image(digit, os.path.join(output_dir, f"generated_digit_{i}.png"))

    # Save a grid of images
    utils.save_image(
        generated_digits, os.path.join(output_dir, "generated_digits_grid.png"), nrow=5
    )

    print(f"Generated digits saved in {output_dir}")


# Generate and save digits
save_generated_digits(num_samples=25)


# Export model components to ONNX
def export_to_onnx(model, output_dir="onnx_models"):
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("ONNX or ONNX Runtime is not installed. Please install them using:")
        print("pip install onnx onnxruntime")
        print("Then run this function again.")
        return

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Export full model
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_dir}/vae_full.onnx",
        input_names=["input"],
        output_names=["reconstructed", "mu", "logvar"],
    )

    # Export encoder
    class EncoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            return self.vae.encode(x.view(-1, 784))

    encoder_wrapper = EncoderWrapper(model)
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        f"{output_dir}/vae_encoder.onnx",
        input_names=["input"],
        output_names=["mu", "logvar"],
    )

    # Export reparameterize
    class ReparameterizeWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, mu, logvar):
            return self.vae.reparameterize(mu, logvar)

    reparameterize_wrapper = ReparameterizeWrapper(model)
    dummy_mu = torch.randn(1, latent_dim, device=device)
    dummy_logvar = torch.randn(1, latent_dim, device=device)
    torch.onnx.export(
        reparameterize_wrapper,
        (dummy_mu, dummy_logvar),
        f"{output_dir}/vae_reparameterize.onnx",
        input_names=["mu", "logvar"],
        output_names=["z"],
    )

    # Export decoder
    class DecoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z):
            return self.vae.decode(z)

    decoder_wrapper = DecoderWrapper(model)
    dummy_z = torch.randn(1, latent_dim, device=device)
    torch.onnx.export(
        decoder_wrapper,
        dummy_z,
        f"{output_dir}/vae_decoder.onnx",
        input_names=["z"],
        output_names=["reconstructed"],
    )

    print(f"Model components exported to ONNX format in directory: {output_dir}")

    # Verify exported models
    for model_name in [
        "vae_full.onnx",
        "vae_encoder.onnx",
        "vae_reparameterize.onnx",
        "vae_decoder.onnx",
    ]:
        onnx_model = onnx.load(f"{output_dir}/{model_name}")
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model {model_name} checked successfully!")


# Export the model components to ONNX
export_to_onnx(model)


# Function to run inference on ONNX models
def onnx_inference(model_path, inputs):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[i].name: inp for i, inp in enumerate(inputs)}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs


# Example usage of ONNX models
print("\nRunning inference on ONNX models:")
dummy_input = torch.randn(1, 1, 28, 28).numpy()
dummy_mu = torch.randn(1, latent_dim).numpy()
dummy_logvar = torch.randn(1, latent_dim).numpy()
dummy_z = torch.randn(1, latent_dim).numpy()

print("Full VAE inference:")
full_outputs = onnx_inference("onnx_models/vae_full.onnx", [dummy_input])
print(
    f"  Outputs: reconstructed shape: {full_outputs[0].shape}, mu shape: {full_outputs[1].shape}, logvar shape: {full_outputs[2].shape}"
)

print("Encoder inference:")
encoder_outputs = onnx_inference("onnx_models/vae_encoder.onnx", [dummy_input])
print(
    f"  Outputs: mu shape: {encoder_outputs[0].shape}, logvar shape: {encoder_outputs[1].shape}"
)

print("Reparameterize inference:")
reparameterize_outputs = onnx_inference(
    "onnx_models/vae_reparameterize.onnx", [dummy_mu, dummy_logvar]
)
print(f"  Outputs: z shape: {reparameterize_outputs[0].shape}")

print("Decoder inference:")
decoder_outputs = onnx_inference("onnx_models/vae_decoder.onnx", [dummy_z])
print(f"  Outputs: reconstructed shape: {decoder_outputs[0].shape}")

print("\nONNX inference completed successfully!")
