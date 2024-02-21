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
        "layer-norm": torch.nn.LayerNorm(normalized_shape=normalized_shape),
        # PyTorch default 1-dimensional batch normalization. Needs to transpose
        # embedding and sequence dimension to normalized over the embedding
        # dimension, which is expected to be second.
        "batch-norm": torch.nn.Sequential(
            Transpose(), torch.nn.LazyBatchNorm1d(), Transpose()
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
            in_proj_weight_quant=weight_quantizer(bits=8, _signed=True),
            # Quantize the bias of the input projections as configured
            in_proj_bias_quant=bias_quantizer(bits=8, _signed=True),
            # No quantization in front of the input projections as this is
            # either done by the output quantization of the preceding attention
            # block (see norm_mlp) or in case of the first layer by the separate
            # input quantization
            in_proj_input_quant=None,

            # Quantize the output projections weights as configured
            out_proj_weight_quant=weight_quantizer(bits=8, _signed=True),
            # Quantize the bias of the output projections as configured
            out_proj_bias_quant=bias_quantizer(bits=8, _signed=True),
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
            # Quantize the LayerNorm outputs
            QuantIdentity(
                # Quantize at the output
                act_quant=act_quantizer(bits, _signed=True),
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            )
        )

        # Quantized MLP following the scaled dot-product attention
        self.mlp = torch.nn.Sequential(
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
                weight_quant=weight_quantizer(bits=8, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits=8, _signed=True),
                # No input quantizer as this is directly preceded by the output
                # quantizer of the norm layer above, following the attention
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
                weight_quant=weight_quantizer(bits=8, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits=8, _signed=True),
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
            # Quantize the LayerNorm outputs
            QuantIdentity(
                # Quantize at the output
                act_quant=act_quantizer(bits, _signed=True),
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            )
        )

    # Forward pass through the transformer block
    def forward(self, x):
        # Scaled dot-product attention with residual branch and normalization
        # Note: No attention mask for now
        x = self.norm_sdp(self.residual_sdp(x, self.sdp(x, x, x)[0]))
        # MLP layer with residual branch and normalization
        return self.norm_mlp(self.residual_mlp(x, self.mlp(x)))


# Quantized binary positional encoding layer
class QuantBinaryPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
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
            [(x & (1 << bit)) >> bit for bit in range(emb)] for x in range(seq)
        ])
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device)
        # Add the quantized encoding tp the quantized input
        #   Note: Convert encoding to bipolar representation
        return self.add(x, 2 * pos - 1)


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
            # Quantization bit-width at the model inputs: Typically this should
            # be higher than for most other layers, e.g., keep this at 8 bits
            input_bits=8,
            # Quantization bit-width at the model outputs: Typically this should
            # be higher than for most other layers, e.g., keep this at 8 bits
            output_bits=8
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Model input quantization is configured separately form all other
        # quantizers
        self.input_quant = QuantIdentity(
            # Quantize at the output of the Identity function
            act_quant=act_quantizer(input_bits, _signed=True),
            # Pass quantization information on to the next layer
            return_quant_tensor=True
        )

        # Positional encoding layer at the input
        self.pos = QuantBinaryPositionalEncoding(
            # Quantize the inputs to the positional encoding to the same
            # bit-with as the input
            input_quant=act_quantizer(input_bits, _signed=True),
            # Pass quantization information on to the next layer
            return_quant_tensor=True
        )

        # Sequence of num_layers transformer encoder blocks
        self.encoder = torch.nn.Sequential(*[
            TransformerBlock(
                num_heads, emb_dim, mlp_dim, bias, norm, dropout, bits
            ) for _ in range(num_layers)
        ])

        # Global average pooling along the sequence dimension
        class GlobalAveragePool(torch.nn.Module):
            # Forward pass averaging the feature map
            def forward(self, x):  # noqa: May be static
                # Compute mean along the sequence dimension, which for
                # batch-first layout is dim=1
                return torch.mean(x, dim=1, keepdim=False)

        # Classification head attached at the end
        self.cls_head = torch.nn.Sequential(
            GlobalAveragePool(),
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
                # Quantize the input of the layer, there is no output
                # quantization in the preceding pooling layer
                input_quant=act_quantizer(output_bits, _signed=True),
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
        x = self.pos(self.input_quant(x))
        # Apply the classification head to the output of the sequence of
        # transformer encoder layers
        return self.cls_head(self.encoder(x))
