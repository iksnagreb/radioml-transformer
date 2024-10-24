# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Base class for all QONNX graph transformations and some basic cleanup
# transformations
from qonnx.transformation.general import (
    Transformation,
    ConvertDivToMul,
    ConvertSubToAdd,
)

# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine

# Groups node inputs by dynamic vs. initializer category
from finn.transformation.streamline.absorb import group_inputs_by_category


# FINN streamlining transformations converting and rounding values
from finn.transformation.streamline import (
    ConvertSignToThres,
    RoundAndClipThresholds
)
# FINN streamlining transformations reordering the graph
from finn.transformation.streamline.reorder import (
    MoveMulPastFork,
    MoveLinearPastFork,
    MoveTransposePastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants,
    MoveTransposePastEltwise,
    MoveMulPastMaxPool,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastMatMul,
    MoveScalarMulPastConv,
    MoveTransposePastJoinMul,
    MoveTransposePastJoinAdd,
    MoveMulPastJoinAdd,
    MoveAddPastJoinAdd,
    MoveScalarLinearPastSplit,
    MoveAffinePastJoinConcat,
    MoveMulPastJoinConcat,
    MoveAddPastJoinConcat,
    MoveTransposePastSplit,
    MoveTransposePastJoinConcat,
    MoveSqueezePastMultiThreshold
)
# FINN streamlining transformations absorbing tensors/nodes into others
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbMulIntoMultiThreshold,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbTransposeIntoMultiThreshold
)
# FINN streamlining transformations fusing/collapsing operations of the same
# kind
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedTranspose,
    CollapseRepeatedAdd
)
# FINN streamlining transformations removing nodes without real effect from the
# graph
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
)

# Custom transformation for exhaustively composing transformations
from .composed_transformation import ComposedTransformation



# Moves constant elementwise multiplication past another joining multiplication
class MoveConstMulPastJoinMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to Multiplications
                if successor.op_type in {"Mul"}:
                    # Applies only if the second multiplication is a join-node
                    if model.is_join_node(successor):
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Need to match the correct input of the joining second
                        # multiplication
                        for i, name in enumerate(successor.input):
                            # If the successors input currently matches the
                            # intermediate tensors, this input needs to be
                            # rewired
                            if name == mid:
                                # Rewire the graph to feed original into the
                                # second Mul node first
                                successor.input[i] = inp
                                # Note: Do not break here as it is perfectly
                                # legal to connect the same tensor multiple
                                # times to different inputs
                        # Repurpose the middle tensor for the output of the
                        # second Mul
                        successor.output[0] = mid
                        # The first Mul operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # The first Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves elementwise multiplication past elementwise addition if one input to
# each of the operators is a known constant
# Note: Reverse of MoveAddPastMul
class MoveMulPastAdd(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to additions
                if successor.op_type in {"Add"}:
                    # The addition may not join as we need to know the second
                    # input
                    if not model.is_join_node(successor):
                        # Get the constant initializer tensors for both
                        # operations: y = s * x + b
                        _, s_name = group_inputs_by_category(node, model)
                        _, b_name = group_inputs_by_category(successor, model)
                        # Skip if either node has no constant initializer
                        if not s_name or not b_name:
                            # Skip without warning ok?
                            continue
                        # There must be exactly one constant per operations
                        assert len(s_name) == 1, \
                            f"To many constant inputs for {node}"
                        assert len(b_name) == 1, \
                            f"To many constant inputs for {successor}"
                        # Now read the initializer tensors
                        s = model.get_initializer(*s_name)
                        b = model.get_initializer(*b_name)
                        # Update the addition initializer according to the
                        # distributive law
                        model.set_initializer(*b_name, b / s)
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Rewire the graph to feed original input into the
                        # Add node first
                        successor.input[0] = inp
                        # Repurpose the middle tensor for the output of the Add
                        successor.output[0] = mid
                        # The Mul operator now gets the middle tensor as its
                        # input
                        node.input[0] = mid
                        # Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Define a set of custom streamlining transformations: These are applied once
# during the actual streamlining step and once after converting attention to
# hardware (the associated cleanup afterward might enable some Streamlining
# transformations once again)
def Streamline():  # noqa: Uppercase
    # Return a set of exhaustively applies transformations
    return ComposedTransformation([
        # On skip-connections: prefer pushing scalar multiplication forward
        # before MoveAddPastMul
        MoveMulPastFork(),
        # The "standard" set of FINN streamlining transformations or at least
        # inspired by them but applied exhaustively until none of them changes
        # the graph anymore.
        # Note: Covers most parts of non-branching linear topologies
        ComposedTransformation([
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
        ]),
        # Streamlining scales and biases forward through residual topologies
        # Note: This mostly covers forking and joining operations
        ComposedTransformation([
            # Note: This is probably the most common way of joining skip
            # connections, i.e., this corresponds to the original residual
            # addition, i.e., y = f(x) + x
            MoveLinearPastEltwiseAdd(),
            MoveLinearPastFork(),
            MoveScalarLinearPastInvariants(),
            MoveMulPastFork(),
            MoveMulPastJoinAdd(),
            MoveAddPastJoinAdd(),
            # Note: This brings constant Muls (i.e., quantizer scales to be
            # removed) forward through joining Muls (i.e., those ending up
            # as actual hardware operators).
            MoveConstMulPastJoinMul()
        ]),
        # Streamlining scales and biases forward through shape/layout changing
        # operations, i.e., mostly transposes
        ComposedTransformation([
            # Streamlining for Split and Concat operations
            MoveScalarLinearPastSplit(),
            MoveAffinePastJoinConcat(),
            MoveMulPastJoinConcat(),
            MoveAddPastJoinConcat(),
            # Move transposes around to some place where they could be removed
            # later, i.e., where they collapse into identities
            MoveTransposePastFork(),
            MoveTransposePastSplit(),
            MoveTransposePastJoinConcat(),
            MoveTransposePastEltwise(),
            MoveTransposePastJoinMul(),
            MoveTransposePastJoinAdd(),
            CollapseRepeatedTranspose(),
            # Remove identity shape/layout transformations
            RemoveIdentityTranspose(),
            RemoveIdentityReshape(),
            # Squeeze operators can be moved past the thresholding
            MoveSqueezePastMultiThreshold(),
            # A certain type of 4d-layout transpose can be absorbed (actually
            # moved past) MultiThreshold operations
            AbsorbTransposeIntoMultiThreshold(),
        ]),
        # Only round and clip after all streamlining transformations have
        # been applied exhaustively.
        # Note: Might still enable another round of streamlining.
        RoundAndClipThresholds(),
    ])
