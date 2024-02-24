# System functionality like creating directories
import os
# YAML for saving experiment metrics
import yaml
# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar in for loops
import tqdm
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader

# The RadioML modulation classification transformer model
from model import RadioMLTransformer
# The RadioML modulation classification dataset
from dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed


# Gets an optimizer instance according to configuration and register the model
# parameters
def get_optimizer(algorithm, parameters, **kwargs):
    # Supported optimizer algorithms
    algorithms = {
        "adam": torch.optim.Adam, "sgd": torch.optim.SGD
    }
    # The configured algorithm must be among the supported ones
    assert algorithm in algorithms, f"Optimizer {algorithm} is not supported"
    # Create the optimizer instance forwarding additional arguments and
    # registering the model parameters
    return algorithms[algorithm](parameters, **kwargs)


# Gets the loss functions instance according to configuration
def get_criterion(criterion, **kwargs):
    # Supported optimization criteria
    criteria = {
        "cross-entropy": torch.nn.CrossEntropyLoss, "nll": torch.nn.NLLLoss
    }
    # The configured criterion must be among the supported ones
    assert criterion in criteria, f"Criterion {criterion} is not supported"
    # Create the loss function instance forwarding additional arguments
    return criteria[criterion](**kwargs)


# Main training loop: Takes a model, loads the dataset and sets up the
# optimizer. Runs the configured number of training epochs
def train(
        model, dataset, batch_size, epochs, criterion, optimizer, loader  # noqa
):
    # Check whether GPU training is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to training mode
    model = model.train()  # noqa: Shadows model...

    # Get the optimizer and register the model parameters
    optimizer = get_optimizer(  # noqa
        **optimizer, parameters=model.parameters()
    )
    # Get the optimization criterion instance
    criterion = get_criterion(criterion)

    # Load the RadioML dataset splits as configured
    train_data, valid_data, _ = get_datasets(**dataset)
    # Create a batched and shuffled data loader for each of the dataset splits
    train_data = DataLoader(train_data, batch_size=batch_size, **loader)
    valid_data = DataLoader(valid_data, batch_size=batch_size, **loader)

    # Collect training and validation loss in a dictionary
    log = {"loss": []}  # noqa: Shadows log from outer scope

    # Run the configured number of training epochs
    for _ in tqdm.trange(epochs, desc="epoch"):
        # Collect training and validation loss per epoch
        train_loss, valid_loss = (0, 0)
        # Set model to training mode
        model = model.train()  # noqa: Shadows model...
        # Iterate the batches of (input, target labels) pairs
        # Note: Ignore the signal-to-noise ratio, third in tuple
        for x, y, _ in tqdm.tqdm(train_data, desc="train-batch", leave=False):
            # Clear gradients of last iteration
            optimizer.zero_grad(set_to_none=True)
            # Feed input data to model to get predictions
            p = model(x.to(device))  # noqa: Duplicate, see below
            # Repeat the target class labels for each sequence step
            y = torch.stack(x.shape[1] * [y], dim=-1).flatten()
            # Loss between class probabilities and true class labels
            loss = criterion(p.reshape(y.shape[0], -1), y.to(device))
            # Backpropagation of the error to compute gradients
            loss.backward()
            # Parameter update step
            optimizer.step()
            # Accumulate the loss over the whole validation dataset
            train_loss += loss.item()
        # Clear gradients of last iteration
        optimizer.zero_grad(set_to_none=True)
        # Switch the model to evaluation mode, disabling dropouts and scale
        # calibration
        model = model.eval()  # noqa: Shadows model...
        # Validation requires no gradients
        with torch.no_grad():
            # Iterate the batches of (input, target labels) pairs
            # Note: Ignore the signal-to-noise ratio, third in tuple
            for x, y, _ in tqdm.tqdm(valid_data, "valid-batch", leave=False):
                # Feed input data to model to get predictions
                p = model(x.to(device))  # noqa: Duplicate, see above
                # Repeat the target class labels for each sequence step
                y = torch.stack(x.shape[1] * [y], dim=-1).flatten()
                # Loss between class probabilities and true class labels
                loss = criterion(p.reshape(y.shape[0], -1), y.to(device))
                # Accumulate the loss over the whole validation dataset
                valid_loss += loss.item()
        # Append loss information to the log
        log["loss"].append({"train": train_loss, "valid": valid_loss})
    # Clear the gradients of last iteration
    optimizer.zero_grad(set_to_none=True)
    # Return the model, the optimizer state and the log after training
    return model, optimizer, log


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml", stages="train")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Pass the model and the training configuration to the training loop
    model, optimizer, log = train(
        model, dataset=params["dataset"], **params["train"]
    )
    # Create the output directory if it does not already exist
    os.makedirs("outputs/", exist_ok=True)
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "outputs/model.pt")
    # Save the optimizer state in PyTorch format
    torch.save(optimizer.state_dict(), "outputs/optimizer.pt")
    # Save the training loss log as YAML
    with open("loss.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(log, file)
