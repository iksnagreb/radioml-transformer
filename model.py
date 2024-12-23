# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas: Quantized versions of PyTorch layers
from brevitas.nn import (
    QuantMultiheadAttention,
    QuantEltwiseAdd,
    QuantIdentity,
    QuantLinear,
    QuantReLU
)


# Derives a weight quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def weight_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant.base import NarrowIntQuant, MaxStatsScaling
    from brevitas.quant.solver import WeightQuantSolver
    from brevitas.inject.enum import RestrictValueType

    # Derive a Quantizer from the brevitas bases
    class Quantizer(NarrowIntQuant, MaxStatsScaling, WeightQuantSolver):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


# Derives a bias quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def bias_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant import IntBias

    # Derive a Quantizer from the brevitas bases
    class Quantizer(IntBias):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Do not require the bit-width to be adjusted to fit the accumulator to
        # which the bias is added
        requires_input_bit_width = False

    # Return the derived quantizer configuration
    return Quantizer


# Derives an activation quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def act_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant.base import IntQuant, ParamFromRuntimePercentileScaling
    from brevitas.quant.solver import ActQuantSolver
    from brevitas.inject.enum import RestrictValueType

    # Derive a Quantizer from the brevitas bases
    class Quantizer(
        IntQuant, ParamFromRuntimePercentileScaling, ActQuantSolver
    ):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


# Gets the normalization layer from configuration key
def get_norm(key, normalized_shape):
    # Transposes Sequence and Embedding dimensions
    class Transpose(torch.nn.Module):
        # Forward pass transposing the feature map
        def forward(self, x):  # noqa: May be static
            # Transpose the last two dimensions of batch x seq x emb layout
            return torch.transpose(x, dim0=-1, dim1=-2)

    # Dictionary mapping keys to supported normalization layer implementations
    norms = {
        # PyTorch default layer normalization. Needs to know the shape of the
        # feature map to be normalized
        "layer-norm": torch.nn.LayerNorm(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            normalized_shape=normalized_shape, elementwise_affine=False
        ),
        # PyTorch default 1-dimensional batch normalization. Needs to transpose
        # embedding and sequence dimension to normalized over the embedding
        # dimension, which is expected to be second.
        "batch-norm": torch.nn.Sequential(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            Transpose(), torch.nn.LazyBatchNorm1d(affine=False), Transpose()
        ),
        # No normalization by a PyTorch built-in identity layer. Should not
        # appear in the graph.
        "none": torch.nn.Identity()
    }

    # Select the normalization layer by key
    return norms[key]


# Single-layer scaled dot-product attention block with MLP and normalization
class TransformerBlock(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, num_heads, emb_dim, mlp_dim, bias, norm, dropout, bits):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Input quantizer to the scaled dot-product attention operations, shared
        # by queries, keys and values inputs. It is important to have this
        # quantizer separate and not preceding the fork node of the residual
        # branches to avoid consecutive quantizers in the skip branch.
        # Note: For some reason it seems not to be possible to use the
        #   in_proj_input_quant of the attention operator
        self.sdp_input_quant = QuantIdentity(
            # Quantize at the output
            act_quant=act_quantizer(bits, _signed=True),
            # Pass quantization information on to the next layer.
            return_quant_tensor=True
        )
        # Quantized scaled dot-product attention operator
        self.sdp = QuantMultiheadAttention(
            # Size of the embedding dimension (input and output)
            embed_dim=emb_dim,
            # Number of attention heads
            num_heads=num_heads,
            # Enable a bias added to the input and output projections
            bias=bias,
            # Layout of the inputs:
            #   Batch x Sequence x Embedding (batch-first, True)
            #   Sequence x Batch x Embedding (batch-second, False)
            batch_first=True,
            # Amount of dropout to apply at the attention block output, i.e.,
            # after the output projection, during training
            dropout=dropout,
            # If query, key and value input are the same, packed input
            # projections use a single, large linear projection to produce
            # the actual query, key and value inputs. Otherwise, use
            # separate linear projections on each individual input.
            packed_in_proj=False,
            # Brevitas has this as an unsigned quantizer by default, but
            # finn can only handle signed quantizer
            attn_output_weights_quant=act_quantizer(bits, _signed=True),
            # Insert an additional quantizer in front ot the softmax. In our
            # finn custom-op, this will be matched to the quantizer
            # following the query and key matmul.
            softmax_input_quant=act_quantizer(bits, _signed=True),
            # Quantize the input projections weights as configured
            in_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the input projections as configured
            in_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # No quantization in front of the input projections as this is
            # either done by a standalone quantizer preceding the whole block
            in_proj_input_quant=None,

            # Quantize the output projections weights as configured
            out_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the output projections as configured
            out_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # Quantize the input to the output projection as configured
            out_proj_input_quant=act_quantizer(bits, _signed=True),

            # Quantizer the key after projections as configured
            k_transposed_quant=act_quantizer(bits, _signed=True),
            # Quantize the queries after projections as configured
            q_scaled_quant=act_quantizer(bits, _signed=True),
            # Quantize the values after projection as configured
            v_quant=act_quantizer(bits, _signed=True),

            # No output quantization for now, as stacking multiple layers
            # results in multiple multi-thresholds in succession
            out_proj_output_quant=None,

            # Return the quantization parameters so the next layer can
            # quantize the bias
            return_quant_tensor=True
        )
        # Residual branch addition skipping over the attention layer
        self.residual_sdp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            return_quant_tensor=True
        )
        # Normalization following the attention layer
        self.norm_sdp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the MLP residual
            # branch. See input quantizer in front of the first MLP layer.
        )

        # Quantized MLP following the scaled dot-product attention
        self.mlp = torch.nn.Sequential(
            # Quantize the inputs to the MLP block. Placed here to not have this
            # at the input of the residual branch.
            QuantIdentity(
                # Quantize at the output
                act_quant=act_quantizer(bits, _signed=True),
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            ),
            # First mlp layer projecting to the mlp dimension
            QuantLinear(
                # Inputs have the size of the attention embedding dimension
                emb_dim,
                # Project to the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as this is directly preceded by a
                # standalone quantizer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized ReLU activation taking care of quantization
                output_quant=None,
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            QuantReLU(
                # Note: ReLU must be quantized to unsigned representation
                act_quant=act_quantizer(bits, _signed=False),
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # Amount of dropout to apply at the sublayer output
            torch.nn.Dropout(p=dropout),
            # Second mlp layer projecting back to the embedding dimension
            QuantLinear(
                # Inputs have the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Project back to the size of the attention embedding dimension
                emb_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as the inputs are already quantized by the
                # preceding ReLU layer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized element-wise addition taking care of quantization
                output_quant=None,
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            ),
            # Amount of dropout to apply at the sublayer output
            torch.nn.Dropout(p=dropout)
        )
        # Residual branch addition skipping over the MLP layer
        self.residual_mlp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            # Note: Not for the last layer to allow this to be combined with
            # standard pytorch calls like .detach() or .numpy(), which are
            # not directly available on QuantTensor.
            return_quant_tensor=True
        )
        # Normalization following the attention layer
        self.norm_mlp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the SDP residual
            # branch
        )

    # Forward pass through the transformer block
    def forward(self, x):
        # Quantize the input to the attention block
        q = self.sdp_input_quant(x)
        # Scaled dot-product attention with residual branch and normalization
        # Note: No attention mask for now
        x = self.norm_sdp(self.residual_sdp(x, self.sdp(q, q, q)[0]))
        # MLP layer with residual branch and normalization
        return self.norm_mlp(self.residual_mlp(x, self.mlp(x)))


# Quantized sinusoidal positional encoding layer
class QuantSinusoidalPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Get the size of the inputs to dynamically generate encodings of the
        # same size
        _, seq, emb = x.shape
        # Start by enumerating all steps of the sequence
        i = torch.as_tensor([[n] for n in range(seq)])
        # Scale factor adjusting the frequency/wavelength of the sinusoid
        # depending on the embedding dimension index
        f = torch.as_tensor([1e4 ** -(i / emb) for i in range(0, emb, 2)])
        # Prepare empty positional encoding tensor of the same size as the input
        pos = torch.empty(seq, emb)
        # Fill the positional encoding with alternating sine and cosine waves
        pos[:, 0::2] = torch.sin(f * i)
        pos[:, 1::2] = torch.cos(f * i)
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=x.dtype)
        # Add the quantized encoding to the quantized input
        return self.add(x, pos)


# Quantized learned positional encoding layer
class QuantLearnedPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(
            self,
            seq_len,
            emb_dim,
            input_quant,
            output_quant,
            return_quant_tensor
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )
        # Register a parameter tensor representing the not quantized positional
        # encoding
        self.pos = torch.nn.Parameter(torch.empty(seq_len, emb_dim))
        # Reset/Initialize the parameter tensor
        self.reset_parameters()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        # Initialize the positional encoding from a normal distribution with
        # zero mean and unit standard deviation
        torch.nn.init.normal_(self.pos, mean=0, std=1)

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Add the quantized encoding to the quantized input
        return self.add(x, self.pos)


# Lazy version of the learned encoding not requiring input dimensions at
# initialization, inferring these at the first forward pass
class LazyQuantLearnedPositionalEncoding(
    torch.nn.modules.lazy.LazyModuleMixin, QuantLearnedPositionalEncoding # noqa
):
    # Once initialized, this will become a QuantLearnedPositionalEncoding as
    # defined above
    cls_to_become = QuantLearnedPositionalEncoding
    # Parameter tensor of the QuantLearnedPositionalEncoding is uninitialized
    pos: torch.nn.UninitializedParameter

    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the quantizer parts of QuantLearnedPositionalEncoding,
        # leaving the dimensions empty
        super().__init__(0, 0, input_quant, output_quant, return_quant_tensor)
        # Register an uninitialized parameter tensor for the positional encoding
        self.pos = torch.nn.UninitializedParameter()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        # If this has already been initialized, delegate to the actual
        # implementation
        if not self.has_uninitialized_params():
            super().reset_parameters()

    # Initializes/Materializes the uninitialized parameter tensor given some
    # sample input tensor to infer the dimensions
    def initialize_parameters(self, x):
        # Only materialize the parameter tensor if it is not yet initialized
        if self.has_uninitialized_params():
            # Do not accumulate gradient information from initialization
            with torch.no_grad():
                # Get the size of the inputs to generate encodings of the same
                # size
                _, seq, emb = x.shape
                # Materialize the positional encoding parameter tensor
                self.pos.materialize((seq, emb))
                # Properly initialize the parameters by resetting the values
                self.reset_parameters()


# Quantized binary positional encoding layer
class QuantBinaryPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Get the size of the inputs to dynamically generate encodings of the
        # same size
        _, seq, emb = x.shape
        # Binary positional encoding fills the embedding dimension with the bit
        # pattern corresponding to the position in the sequence
        pos = torch.as_tensor([
            [(n & (1 << bit)) >> bit for bit in range(emb)] for n in range(seq)
        ])
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=x.dtype)
        # Add the quantized encoding tp the quantized input
        #   Note: Convert encoding to bipolar representation
        return self.add(x, 2 * pos - 1)


# Gets the positional encoding layer from configuration key, quantizers and
# shape
def get_positional_encoding(
        key, input_quant, output_quant, return_quant_tensor
):
    # Dictionary mapping keys to supported normalization layer implementations
    masks = {
        # No positional encoding
        "none": QuantIdentity(
            act_quant=input_quant, return_quant_tensor=return_quant_tensor
        ),
        # Fixed, sinusoidal positional encoding according to Vaswani et al. with
        # added quantizers
        "sinusoidal": QuantSinusoidalPositionalEncoding(
            input_quant, output_quant, return_quant_tensor
        ),
        # Fixed, binary positional encoding with quantizers
        "binary": QuantBinaryPositionalEncoding(
            input_quant, output_quant, return_quant_tensor
        ),
        # Learned positional encoding with quantizers
        "learned": LazyQuantLearnedPositionalEncoding(
            input_quant, output_quant, return_quant_tensor
        )
    }
    # Select the positional encoding type by key
    return masks[key]


# RadioML modulation classification transformer encoder model
class RadioMLTransformer(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(
            self,
            # Number of layers of attention blocks
            num_layers,
            # Number of attention heads per block
            num_heads,
            # Size of embedding dimension going into/out of the attention block
            emb_dim,
            # Size of MLP dimension in each attention block
            mlp_dim,
            # Number of output classes to predict at the output
            num_classes,
            # Enables bias term added to Linear layers
            bias,
            # Quantization bit-width: For now all layers are quantized to the
            # same bit-width
            bits,
            # Dropout: probability of an element to be zeroed during training
            dropout=0.1,
            # Type of normalization layer to use in the transformer blocks
            #   Options are: layer-norm, batch-norm and none
            norm="layer-norm",
            # Type of positional encoding to use at the input
            #   Options are: none, sinusoidal, binary, learned
            positional_encoding="none",
            # Quantization bit-width at the model inputs: Typically this should
            # be higher than for most other layers, e.g., keep this at 8 bits
            input_bits=8,
            # Quantization bit-width at the model outputs: Typically this should
            # be higher than for most other layers, e.g., keep this at 8 bits
            output_bits=8
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Positional encoding layer at the input
        self.pos = get_positional_encoding(
            # Select the implementation by configuration key
            key=positional_encoding,
            # Quantize the inputs to the positional encoding to the same
            # bit-width as the input
            input_quant=act_quantizer(input_bits, _signed=True),
            # Quantize the sum of input and positional encoding to the same
            # bit-width as the input
            output_quant=None,
            # Pass quantization information on to the next layer
            return_quant_tensor=True
        )

        # Sequence of num_layers transformer encoder blocks
        self.encoder = torch.nn.Sequential(*[
            TransformerBlock(
                num_heads, emb_dim, mlp_dim, bias, norm, dropout, bits
            ) for _ in range(num_layers)
        ])

        # Classification head attached at the end
        self.cls_head = torch.nn.Sequential(
            # Project from embedding dimension to the number of classes
            QuantLinear(
                # Inputs have the size of the attention embedding dimension
                emb_dim,
                # Project to the configured number of classes
                num_classes,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=weight_quantizer(output_bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(output_bits, _signed=True),
                # Quantize the inputs to the classification head to the same
                # bit-width as the trunk of the model
                input_quant=act_quantizer(bits, _signed=True),
                # Model output quantization is configured separately form all
                # other quantizers
                output_quant=act_quantizer(output_bits, _signed=True),
                # Do not return the quantization parameters as this is the last
                # layer of the model
                return_quant_tensor=False
            ),
            # # Softmax normalization to yield class probabilities
            # # Note: Not required when training with cross-entropy loss
            # torch.nn.Softmax(dim=-1)
        )

    # Model forward pass taking an input sequence and returning a single set of
    # class probabilities
    def forward(self, x):
        # Add positional encoding to the input
        x = self.pos(x)
        # Apply the classification head to the output of the sequence of
        # transformer encoder layers
        return self.cls_head(self.encoder(x))
