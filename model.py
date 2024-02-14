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


# Single-layer scaled sot-product attention block with MLP and normalization
class TransformerBlock(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, num_heads, emb_dim, mlp_dim, bias, bits):
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
            softmax_input_quant=None,
            # Quantize the input projections weights as configured
            in_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the input projections as configured
            in_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # No quantization in front of the input projections
            in_proj_input_quant=act_quantizer(bits, _signed=True),

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
            # Vanilla PyTorch LayerNorm without quantization
            torch.nn.LayerNorm(normalized_shape=emb_dim),
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
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # Quantize the input of the layer
                input_quant=None,
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
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            )
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
            # Vanilla PyTorch LayerNorm without quantization
            torch.nn.LayerNorm(normalized_shape=emb_dim),
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


# RadioML fingerprinting transformer encoder model
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
            bits
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Sequence of num_layers transformer encoder blocks
        self.encoder = torch.nn.Sequential(*[
            TransformerBlock(
                num_heads, emb_dim, mlp_dim, bias, bits
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
            # Perform global average pooling along the sequence length
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
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # Quantize the input of the layer
                input_quant=act_quantizer(bits, _signed=True),
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # # Softmax normalization to yield class probabilities
            # torch.nn.Softmax(dim=-1)
        )

    # Model forward pass taking an input sequence and returning a single set of
    # class probabilities
    def forward(self, x):
        # Apply the classification head to the output of the sequence of
        # transformer encoder layers
        return self.cls_head(self.encoder(x))
