# YAML for loading experiment configurations
import yaml
# Save verification input-output pair as numpy array
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas to QONNX model export
from brevitas.export import export_qonnx

# The RadioML modulation classification transformer model
from model import RadioMLTransformer
# The RadioML modulation classification dataset
from dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, **kwargs):  # noqa: Shadows model
    # Do the forward pass for generating verification data and tracing the model
    # for export on CPU only
    device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...

    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(**dataset)
    # Sample a single input from the dataset
    inp, _, _ = eval_data[0]
    # Convert the input tensor from Numpy to PyTorch format and add batch
    # dimension
    inp = torch.as_tensor([inp]).to(device)
    # Forward pass of the sample input to generate input-output pair for
    # verification
    out = model(inp)

    # Save the input and output data for verification purposes later
    np.save("outputs/inp.npy", inp.detach().numpy())
    np.save("outputs/out.npy", out.detach().numpy())
    # Export the model graph to QONNX
    export_qonnx(model, (inp,), "outputs/model.onnx", **kwargs)


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/model.pt"))
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], **params["export"])
