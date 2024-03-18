# Use the DVC api for loading the YAML parameters
import dvc.api
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


# Check whether a layer is a normalization layer of some supported type
def is_norm_layer(module):
    # Set of normalization layer (bases) which maybe need to be patched
    norm_layers = {
        # All BatchNorm and InstanceNorm variants derive from this baseclass
        torch.nn.modules.batchnorm._NormBase,  # noqa: Access to _NormBase
        # LayerNorm has a unique implementation
        torch.nn.LayerNorm,
    }
    # Check the module against all supported norm layer types
    return any(isinstance(module, norm) for norm in norm_layers)


# Fixes export issues of normalization layers with disabled affine parameters.
# Somehow the export to ONNX trips when it encounters the weight and bias tensor
# to be 'None'.
def patch_non_affine_norms(model: torch.nn.Module):  # noqa: Shadows model
    # Iterate all modules in the model container
    for name, module in model.named_modules():
        # If the module is a normalization layer it might require patching the
        # affine parameters
        if is_norm_layer(module):
            # Check whether affine scale parameters are missing
            if hasattr(module, "weight") and module.weight is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_var"):
                    # Patch the affine bias by all 1 tensor of the same shape,
                    # type and device as the running variance
                    module.weight = torch.nn.Parameter(
                        torch.ones_like(module.running_var)
                    )
            # Check whether affine bias parameters are missing
            if hasattr(module, "bias") and module.bias is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_mean"):
                    # Patch the affine bias by all 0 tensor of the same shape,
                    # type and device as the running mean
                    module.bias = torch.nn.Parameter(
                        torch.zeros_like(module.running_var)
                    )
    # Return the patched model container
    return model


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml", stages="export")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/model.pt", map_location="cpu"))
    # Prevent export issue for missing affine normalization parameters
    model = patch_non_affine_norms(model)
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], **params["export"])
