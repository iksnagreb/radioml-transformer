# YAML for loading experiment configurations
import yaml
# Progressbar in for loops
import tqdm
# PyTorch base package: Math and Tensor Stuff
import torch

# The RadioML fingerprinting transformer model
from model import RadioMLTransformer


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
def train(model, dataset, batch_size, epochs, criterion, optimizer):  # noqa
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

    # Dummy dataset
    train_data = 100 * [
        (torch.rand(batch_size, 32, 64), torch.randint(0, 14, (32,)))
    ]
    valid_data = train_data

    # Collect training and validation statistics in a dictionary
    log = {"train": [], "valid": []}  # noqa: Shadows log from outer scope

    # Run the configured number of training epochs
    for epoch in tqdm.trange(epochs, desc="epochs"):
        # Collect training and validation loss per epoch
        train_loss, valid_loss = (0, 0)
        # Set model to training mode
        model = model.train()  # noqa: Shadows model...
        # Iterate the batches of (input, target labels) pairs
        for x, y in train_data:
            # Clear gradients of last iteration
            optimizer.zero_grad(set_to_none=True)
            # Feed input data to model to get predictions
            p = model(x.to(device))
            # Loss between class probabilities and true class labels
            loss = criterion(p, y.to(device))
            # Backpropagation of the error to compute gradients
            loss.backward()
            # Parameter update step
            optimizer.step()
            # Accumulate the loss over the whole validation dataset
            train_loss += loss.item()
        # Append loss information to the log
        log["train"].append({"loss": train_loss})
        # Clear gradients of last iteration
        optimizer.zero_grad(set_to_none=True)
        # Switch the model to evaluation mode, disabling dropouts and scale
        # calibration
        model = model.eval()  # noqa: Shadows model...
        # Validation requires no gradients
        with torch.no_grad():
            # Iterate the batches of (input, target labels) pairs
            for x, y in valid_data:
                # Feed input data to model to get predictions
                p = model(x.to(device))
                # Loss between class probabilities and true class labels
                loss = criterion(p, y.to(device))
                # Accumulate the loss over the whole validation dataset
                valid_loss += loss.item()
        # Append loss information to the log
        log["valid"].append({"loss": valid_loss})
    # Clear the gradients of last iteration
    optimizer.zero_grad(set_to_none=True)
    # Return the model, the optimizer state and the log after training
    return model, optimizer, log


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Pass the model and the training configuration to the training loop
    model, optimizer, log = train(model, **params["train"])
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "model.pt")
    # Save the optimizer state in PyTorch format
    torch.save(optimizer.state_dict(), "optimizer.pt")
    # Save the training log as YAML
    with open("log.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(log, file)
