# Interaction with the python interpreter: Inserting path to the deployed driver
import sys
# YAML for loading experiment configurations
import yaml
# Numpy for handling arrays (inputs/outputs to/from the model)
import numpy as np

# Progressbar in for loops
import tqdm
# Pandas to handle the results as table, i.e., DataFrame
import pandas as pd
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader
# Classification metrics
from sklearn.metrics import accuracy_score

# The RadioML modulation classification dataset
from dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed

# For some reason, pynq must be imported here - exactly here, otherwise the
# script fails freeing some allocated buffers at the end (after everything else
# passed successfully)
import pynq  # noqa: This is not really used except for fixing this weird bug

# QONNX wrapper around ONNX models
from qonnx.core.modelwrapper import ModelWrapper
# Convert ONNX nodes to QONNX CustomOp instances
from qonnx.custom_op.registry import getCustomOp


# Main evaluation loop: Takes a trained model, loads the dataset and sets and
# runs a bunch of inferences to collect metrics
def evaluate(model, dataset, batch_size, loader):  # noqa: Shadows model
    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(**dataset)
    # Create a batched and shuffled data loader for each of the dataset splits
    eval_data = DataLoader(eval_data, batch_size=batch_size, **loader)

    # Start collecting predictions and ground-truth in a data frame
    results = []

    # Evaluation requires no gradients
    with torch.no_grad():
        # Iterate the (input, target labels) pairs
        for x, cls, snr in tqdm.tqdm(eval_data, "eval-batch", leave=False):
            # Model forward pass producing logits corresponding to class
            # probabilities. Aggregate the logits (~ class probabilities) along
            # the sequence dimension.
            p = model(x.numpy()).sum(axis=1).argmax(axis=-1)
            # Each sample in the batch will generate a new record
            for cls, p, snr in zip(cls, p, snr):  # noqa: Shadows cls, p, snr
                # Append prediction and ground truth to the data frame
                results.append({
                    "cls": cls.item(), "prediction": p.item(), "snr": snr.item()
                })

    # Convert list of records to pandas data frame to do statistics to it
    results = pd.DataFrame.from_records(results)

    # Compute the classification accuracy over the evaluation dataset
    return accuracy_score(*results[["cls", "prediction"]].T.values)


# Extracts the model and sets up the accelerator from the dataflow parent model
def extract_and_setup_model(parent, accelerator):  # noqa: Shadows outer scope
    # Assumption: The whole graph has three nodes: The MultiThreshold operation
    # quantizing the input, the StreamingDataflowPartition corresponding to the
    # FPGA accelerator and a Mul node de-quantizing the output
    assert len(parent.graph.node), "To many node in the dataflow parent graph"

    # Function wrapping the input quantization as it is described by the model
    def quantize(x):
        # The multi thresholds must be the first node of the graph
        multithreshold = parent.graph.node[0]
        # Check whether this is indeed the thresholding quantization
        assert multithreshold.op_type == "MultiThreshold", \
            f"First node must be MultiThreshold: {multithreshold.name}"
        # Get the quantization thresholds which should be stored as an
        # initializer tensor within the model graph
        thresholds = parent.get_initializer(multithreshold.input[1])
        # Prepare the input execution context
        context = {
            multithreshold.input[0]: x, multithreshold.input[1]: thresholds
        }
        # Execute the node on the input context, writing the result back into
        # the context
        getCustomOp(multithreshold).execute_node(context, parent.graph)
        # Extract the output from the execution context
        return context[multithreshold.output[0]]

    # Function wrapping the output de-quantization as it is described by the
    # model
    def dequantize(x):
        # The de-quantization multiplication node of the graph
        mul = parent.graph.node[2]
        # Check whether this is indeed the mul de-quantization
        assert mul.op_type == "Mul", f"last node must be Mul: {mul.name}"
        # Get the de-quantization scale which should be stored as an initializer
        # tensor within the model graph
        scale = parent.get_initializer(mul.input[1])
        # Apply the de-quantization scale to the tensor
        return scale * x

    # Wrap the whole model as a function
    def model(x):  # noqa: Shadows model from outer scope
        # Chain calls to the quantization/de-quantization and accelerator parts
        return dequantize(accelerator.execute(quantize(x)))

    # Return the model stitching software and accelerator parts in a simple
    # python function interface
    return model


# Adds the batch dimension at the front of a shape
def add_batch(shapes):
    return [(1, *shape) for shape in shapes]


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)

    # Seed all RNGs
    seed(params["seed"])
    # Path to the build output directory
    build = params["build"]["finn"]["output_dir"]
    # Load the parent model of the build dataflow accelerator
    parent = ModelWrapper(f"{build}/intermediate_models/dataflow_parent.onnx")

    # Path to the deployment package generated by FINN
    deploy = f"{build}/deploy/"
    # Add the path to the deployed driver to the search path
    sys.path.append(f"{deploy}/driver")
    # Import the accelerator overlay and configuration from the deployed driver
    from driver import FINNExampleOverlay, io_shape_dict, Device  # noqa

    # Patch the I/O shapes to reintroduce the batch size
    io_shape_dict["ishape_normal"] = add_batch(io_shape_dict["ishape_normal"])
    io_shape_dict["oshape_normal"] = add_batch(io_shape_dict["oshape_normal"])
    io_shape_dict["ishape_folded"] = add_batch(io_shape_dict["ishape_folded"])
    io_shape_dict["oshape_folded"] = add_batch(io_shape_dict["oshape_folded"])
    io_shape_dict["ishape_packed"] = add_batch(io_shape_dict["ishape_packed"])
    io_shape_dict["oshape_packed"] = add_batch(io_shape_dict["oshape_packed"])

    # Load the accelerator overlay
    accelerator = FINNExampleOverlay(
        # Path to the accelerator bitfile built by FINN
        bitfile_name=f"{deploy}/bitfile/finn-accel.bit",
        # Dictionary describing the I/O of the FINN-generated accelerator
        io_shape_dict=io_shape_dict,
        # Path to folder containing runtime-writable .dat weights
        runtime_weight_dir=f"{deploy}/driver/runtime_weights/",
        # Default to the device at index 0 for now...
        device=Device.devices[0],
        # Target platform: zynq-iodma or alveo
        platform="zynq-iodma",
        # The verification pass uses a single sample batch input/output pair
        batch_size=1
    )

    # Extract the software parts of the model and sticht together with the
    # accelerator part
    model = extract_and_setup_model(parent, accelerator)

    # Load the verification input/output pair
    inp = np.load("outputs/inp.npy")
    out = np.load("outputs/out.npy")
    # Run the verification input through the model
    y = model(inp)
    # Compare the output produced by the model to the expected output
    assert np.allclose(y, out), "Produced and expected output do not match"
    # Just print some success message for this simple dummy
    print("Verification on device: SUCCESS")

    # Reset the batch size to the size configured for the evaluation dataset
    accelerator.batch_size = params["eval-on-device"]["batch_size"]
    # Reconfigure the model
    model = extract_and_setup_model(parent, accelerator)

    # Evaluate the model across the evaluation dataset reporting the accuracy
    accuracy = evaluate(
        model, dataset=params["dataset"], **params["eval-on-device"]
    )
    # Print the accuracy report to the standard output
    print(f"Prediction accuracy on evaluation dataset: {accuracy}")
