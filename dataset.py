# Make copies (deep copies) of python objects
import copy
# The RadioML comes in HDF5 format
import h5py
# The data inside the HDF5 file are stored as numpy arrays
import numpy as np
# PyTorch base class for handling datasets
from torch.utils.data import Dataset


# Derive a custom dataset from the PyTorch base
class RadioMLDataset(Dataset):
    # Loads and filters the dataset index
    def __init__(
            self, path, classes=None, signal_to_noise_ratios=None, reshape=None
    ):
        # Open the dataset file in reading mode
        self.file = h5py.File(path, "r")
        # The dataset is composed of three subfiles: the data series, the
        # modulation classes and the signal-to-noise ration of each sample
        data, cls, snr = self.file["X"], self.file["Y"], self.file["Z"]
        # All subfiles must have the same length
        assert len(data) == len(cls) == len(snr), "Dataset is corrupted"

        # Convert one-hot encoded class labels to class numbers and remove
        # dimensions with only 1 element
        self.cls = cls = cls[:].argmax(axis=-1).squeeze()
        # Remove dimensions with only 1 element from the signal-to-noise ratio
        self.snr = snr = snr[:].squeeze()

        # If no classes are specified to be selected, select all classes
        if classes is None:
            # Create a set of all unique classes from the dataset
            classes = {*np.unique(cls)}

        # If no signal-to-noise ratios are specified to be selected, select all
        # available levels
        if signal_to_noise_ratios is None:
            # Create a set of all unique noise levels from the dataset
            signal_to_noise_ratios = {*np.unique(snr)}

        # Filter function deciding whether a sample should be kept according to
        # its class and noise level
        def keep(label, noise):
            # Keep samples which are of the selected classes AND noise levels
            return label in classes and noise in signal_to_noise_ratios

        # Combine class labels and signal-to-noise rations with sample index and
        # filter for the requested classes and noise levels
        self.indices = [
            i for i, (cls, snr) in enumerate(zip(cls, snr)) if keep(cls, snr)
        ]
        # Optional reshaping of the dataset time series
        self.reshape = reshape

    # Get the data sample with class labels and noise level at the index
    def __getitem__(self, idx):
        # Look up the "real" index from the filtered index list
        idx = self.indices[idx]
        # Optional reshaping of the data series. If this is None, numpy
        # reshaping will keep the original shape
        shape = self.reshape
        # Return the (reshaped) data series and label information
        return self.file["X"][idx].reshape(shape), self.cls[idx], self.snr[idx]

    # Length of the filtered dataset
    def __len__(self):
        # Length of the dataset is the length of the index list
        return len(self.indices)

    # Makes a deep copy of the dataset
    def _deepcopy(self):
        # Start by making a normal copy of the object
        _copy = copy.copy(self)
        # Remove the HDF5 file object which cannot be deep-copied
        _copy.file = None
        # Make a proper deep copy of everything else
        _copy = copy.deepcopy(_copy)
        # Reopen the dataset file
        _copy.file = h5py.File(self.file.filename, "r")
        # Return the copied object
        return _copy

    # Generates a train/test split of the dataset
    def split(self, fraction=0.5, seed=0):
        # Initialize a new random number generator with the specified seed to
        # make the splitting reproducible
        rng = np.random.default_rng(seed)
        # Shuffle the index list
        indices = rng.permuted(self.indices)
        # Split the index list
        indices_1, indices_2 = np.split(
            np.asarray(indices), [int(fraction * len(indices))]
        )
        # Make two copies of the dataset
        dataset_1, dataset_2 = self._deepcopy(), self._deepcopy()
        # Replace the index lists of each copy
        dataset_1.indices = indices_1
        dataset_2.indices = indices_2
        # Return both modified copies
        return dataset_1, dataset_2


# Gets the datasets (train/validation/evaluation splits) according to
# configuration
def get_datasets(splits, seed=0, **kwargs):
    # There must be exactly three splits of the dataset
    assert len(splits) == 3, "There must be three splits of the dataset"
    # Splits must sum to 100% of the dataset
    assert sum(splits) == 1, "Splits must cover the whole dataset"

    # Load the RadioML dataset as configured
    dataset = RadioMLDataset(**kwargs)
    # Create the first split into train and validation+evaluation set
    train_data, test_data = dataset.split(fraction=splits[0], seed=seed)
    # Convert the fractions of the second split to be relative to the remaining
    # fraction
    second_fraction = splits[1] / (1.0 - splits[0])
    # Create the second split into validation and evaluation set
    valid_data, eval_data = test_data.split(fraction=second_fraction, seed=seed)
    # Return all three splits of the dataset
    return train_data, valid_data, eval_data
