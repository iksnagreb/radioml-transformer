# YAML for saving experiment metrics
import yaml
# Pandas to handle the reports as table, i.e., DataFrame
import pandas as pd
# For loading the sample input to query the input dimensions
import numpy as np

# FINN dataflow builder
import finn.builder.build_dataflow as build
# FINN dataflow builder configuration
import finn.builder.build_dataflow_config as build_cfg

# Seeding RNGs for reproducibility
from utils import seed

# Custom build steps required to streamline and convert the attention operator
from build_steps import (
    step_tidy_up_pre_attention,
    step_tidy_up_post_attention,
    step_streamline_attention,
    step_streamline_residual,
    step_streamline_norms,
    step_streamline_positional,
    step_convert_attention_to_hw,
    step_convert_elementwise_binary_to_hw,
    step_convert_lookup_to_hw,
    step_replicate_streams,
    set_target_parallelization,
    set_fifo_depths,
    step_apply_folding_config,
    node_by_node_rtlsim,  # noqa: Maybe unused, only for debugging
    node_by_node_cppsim,
)

# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Seed all RNGs
    seed(params["seed"])
    # Extract sequence length and embedding dimension from the verification
    # sample inputs
    _, seq_len, emb_dim = np.load("outputs/inp.npy").shape
    # Create a configuration for building the scaled dot-product attention
    # operator to a hardware accelerator
    cfg = build_cfg.DataflowBuildConfig(
        # Unpack the build configuration parameters
        **params["build"]["finn"],
        # Print all warnings and compiler output to stdout
        verbose=True,
        # Generate and keep the intermediate outputs including reports
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        # Steps after which verification should be run
        verify_steps=[
            # Verify the model after converting to the FINN onnx dialect
            build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
            # Verify the model again using python mode after the default
            # streamlining step
            build_cfg.VerificationStepType.STREAMLINED_PYTHON,
            # Verify the model again after tidy up transformations, right before
            # converting to HLS
            build_cfg.VerificationStepType.TIDY_UP_PYTHON,
            # Verify the model after generating C++ HLS and applying folding
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM
            # No RTL Simulation support for now
        ],
        # File with test inputs for verification
        verify_input_npy="outputs/inp.npy",
        # File with expected test outputs for verification
        verify_expected_output_npy="outputs/out.npy",
        # Save the intermediate model graphs
        save_intermediate_models=True,
        # Avoid RTL simulation for setting the FIFO sizes
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.CHARACTERIZE,
        # Do not automatically set FIFO sizes as this requires RTL simulation
        # not implemented for the attention operator
        auto_fifo_depths=False,
        # Build steps to execute
        steps=[
            # Need to apply some tidy-up transformations before converting to
            # the finn dialect of onnx
            step_tidy_up_pre_attention,
            # Convert all QONNX Quant nodes to Multithreshold nodes
            "step_qonnx_to_finn",
            # Tidy up the graph after converting from QONNX to FINN format
            # Note: Triggers a verification step
            "step_tidy_up",
            # Positional encoding needs to be streamlined first with slightly
            # different order of certain streamlining transformations to avoid
            # weird rounding issue of intermediate results
            step_streamline_positional,
            # Custom streamlining for models containing attention operators
            step_streamline_attention,
            # Streamlining of the residual branches
            step_streamline_residual,
            # Streamline the normalization layers, i.e., transposed batch norm
            step_streamline_norms,
            # Another round using the default streamlining steps
            # Note: Triggers a verification step
            "step_streamline",
            # New conversion of the scaled dot-product attention pattern
            step_convert_attention_to_hw,
            # Another tidy-up step to remove unnecessary dimensions and
            # operations after converting the attention operators to HLS
            step_tidy_up_post_attention,
            # Convert the elementwise binary operations to hardware operators.
            # These include for example adding residual branches and positional
            # encoding
            step_convert_elementwise_binary_to_hw,
            # Convert the Gather layer realizing the input token embedding to
            # the FINN hardware implementation, i.e., the Lookup layer
            step_convert_lookup_to_hw,
            # Properly replicate the stream feeding the query, key and value
            # projections
            step_replicate_streams,
            # Convert most other layers supported by FINN to HW operators
            "step_convert_to_hw",
            # Specialize HW layer implementations as either HLS or RTL
            "step_specialize_layers",
            "step_create_dataflow_partition",
            # Set the folding configuration to meet the cycles per sequence
            # target
            set_target_parallelization(seq_len, emb_dim),
            # Apply folding configuration, specifying hardware implementation
            # details
            # Note: This triggers a verification step
            step_apply_folding_config,
            "step_minimize_bit_width",
            # The ScaledDotProductAttention custom op does not define any
            # estimates
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            # Set the attention- and residual-related FIFO depths insert FIFOs
            # and apply folding configuration once again
            # Note: Implement all FIFOs with a depth at least as deep as the
            # sequence length in URAM.
            set_fifo_depths(seq_len, emb_dim, uram_threshold=seq_len),
            # Run additional node-by-node verification in RTL simulation of the
            # model before creating the stitched IP
            # Note: end-to-end verification of the stitched IP in RTL simulation
            # is still not possible due to missing float IPs
            node_by_node_cppsim,
            # Only for debugging for now, does not work if "vivado" style
            # StreamingFIFOs are used
            # node_by_node_rtlsim,
            "step_create_stitched_ip",
            # Attention does currently not support RTL simulation due to missing
            # float IPs.
            # "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    )

    # Run the build process on the dummy attention operator graph
    build.build_dataflow_cfg("outputs/model.onnx", cfg)

    # Collect and aggregate build metrics like resource utilization
    # Open the report file
    with open(params["build"]["metrics"]["report"]) as file:
        # Load the JSON formatted report
        report = pd.read_json(file, orient="index")
    # Filter the reported rows according to some regex filter rule
    report = report.filter(
        regex=params["build"]["metrics"]["filter"], axis="rows"
    )
    # Generate a summary of the total resources
    summary = report.sum()
    # Dump the metrics dictionary as yaml
    with open("resources.yaml", "w") as file:
        # Convert the dataframe to a dictionary which can be dumped into YAML
        yaml.safe_dump(summary.to_dict(), file)
