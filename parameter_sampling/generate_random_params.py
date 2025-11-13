import sys
import os

TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if TOP_LEVEL_DIR not in sys.path:
    sys.path.insert(0, TOP_LEVEL_DIR)

import numpy as np
import usat_designer.processing.speaker_layouts as sl
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import yaml
from usat_designer.processing.constants import *
from usat_designer.processing.launch_usat import decode_for_random_parameter_generation
from usat_designer.utils import parameter_utils as pu
import usat_designer.utils.directory_utils as dir_utils
import time
import traceback
import argparse
import warnings
import tempfile
import secrets


def parse_from_config(yaml_file):

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    
    # COEFFICIENTS
    coeff_config = config.get(DSN_XML_COEFFICIENTS, {})
    coeffs = {}

    for coeff_name, coeff_data in coeff_config.items():
        coeff_distribution = coeff_data.get(DSN_SMPL_DISTRIBUTION)
        coeff_range        = coeff_data.get(DSN_SMPL_DISTRIBUTION_ARGS)
        coeff_value        = round(get_y_i(coeff_distribution, coeff_range))
        coeffs[coeff_name] = coeff_value

    # FORMATS
    default_ambisonics_order = 1
    
    input_data  = get_random_x_lambda(config[DSN_SMPL_INPUT_FORMAT])
    output_data = get_random_x_lambda(config[DSN_SMPL_OUTPUT_FORMAT])

    encoding_settings = {
        DSN_XML_INPUT_TYPE: input_data[0],
        DSN_XML_OUTPUT_TYPE: output_data[0]
    }

    input_ambisonics    = {DSN_XML_AMBISONICS_ORDER_IN: default_ambisonics_order}
    output_ambisonics   = {DSN_XML_AMBISONICS_ORDER_OUT: default_ambisonics_order}

    input_speaker_layout    = []
    output_speaker_layout   = []
    input_layout_desc       = ""
    output_layout_desc      = ""

    if input_data[0] == DSN_XML_AMBISONICS:
        input_ambisonics[DSN_XML_AMBISONICS_ORDER_IN] = input_data[1]

    elif input_data[0] == DSN_XML_SPEAKER_LAYOUT:
        input_layout_desc = input_data[1]
        input_speaker_layout = sl.SPEAKER_LAYOUTS.get(input_layout_desc)
        if input_speaker_layout is None:
            print(f"Warning: input layout '{input_layout_desc}' not found!")
    else:
        raise ValueError("Input format not supported.")

    if output_data[0] == DSN_XML_AMBISONICS:
        output_ambisonics[DSN_XML_AMBISONICS_ORDER_OUT] = output_data[1]

    elif output_data[0] == DSN_XML_SPEAKER_LAYOUT:
        output_layout_desc      = output_data[1]
        output_speaker_layout   = sl.SPEAKER_LAYOUTS.get(output_layout_desc)
        
        if output_speaker_layout is None:
            print(f"Warning: output layout '{output_layout_desc}' not found!")

    else:
        raise ValueError("Output format not supported.")
    
    return {
        DSN_XML_SETTINGS: encoding_settings,
        DSN_XML_INPUT_AMBISONICS: input_ambisonics,
        DSN_XML_OUTPUT_AMBISONICS: output_ambisonics,
        DSN_XML_INPUT_SPEAKER_LAYOUT: input_speaker_layout,
        DSN_XML_OUTPUT_SPEAKER_LAYOUT: output_speaker_layout,
        DSN_SMPL_INPUT_LAYOUT_DESC: input_layout_desc,
        DSN_SMPL_OUTPUT_LAYOUT_DESC: output_layout_desc,
        DSN_XML_COEFFICIENTS: coeffs
    }

def get_random_x_lambda(config_section):
    formats         = config_section.get(DSN_SMPL_FORMAT_CHOICES)
    selected_format = np.random.choice(formats)

    if selected_format == DSN_XML_AMBISONICS:
        ambisonics_orders   = config_section.get(DSN_XML_AMBISONICS).get(DSN_SMPL_DISTRIBUTION_ARGS)
        selected_value      = np.random.choice(ambisonics_orders)

    elif selected_format == DSN_XML_SPEAKER_LAYOUT:
        layout_names        = config_section.get(DSN_XML_SPEAKER_LAYOUT).get(DSN_SMPL_DISTRIBUTION_ARGS)
        selected_value      = np.random.choice(layout_names)

    else:
        raise ValueError(f"Unsupported format: {selected_format}")

    return selected_format, selected_value


def get_y_i(distribution: str, args) -> float:
    distribution = distribution.lower()
    
    if distribution == "none":
        assert(isinstance(args, float) or isinstance(args, int))
        return args
    
    elif distribution == "uniform":
        low, high = args
        return np.random.uniform(low, high)

    elif distribution == "normal":
        mean, std   = args
        mean        = round(mean, 2)
        std         = round(std, 2)

        sample = -1
        while sample < 0:
            sample = round(np.random.normal(mean, std), 2)
        return sample

    elif distribution == "lognormal":
        mean, sigma = args
        return np.random.lognormal(mean, sigma)

    elif distribution == "beta":
        a, b = args
        return np.random.beta(a, b)

    elif distribution == "choice":
        return np.random.choice(args)

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def build_xml_config(usat_state_parameters):
    state_params_xml            = ET.Element(DSN_XML_USAT_STATE_PARAMETERS)
    encoding_settings_xml       = ET.SubElement(state_params_xml, DSN_XML_SETTINGS)
    input_ambisonics_xml        = ET.SubElement(state_params_xml, DSN_XML_INPUT_AMBISONICS)
    output_ambisonics_xml       = ET.SubElement(state_params_xml, DSN_XML_OUTPUT_AMBISONICS)
    input_speaker_layout_xml    = ET.SubElement(state_params_xml, DSN_XML_INPUT_SPEAKER_LAYOUT)
    output_speaker_layout_xml   = ET.SubElement(state_params_xml, DSN_XML_OUTPUT_SPEAKER_LAYOUT)
    input_layout_desc_xml       = ET.SubElement(state_params_xml, DSN_SMPL_INPUT_LAYOUT_DESC)
    output_layout_desc_xml      = ET.SubElement(state_params_xml, DSN_SMPL_OUTPUT_LAYOUT_DESC)
    coefficients_xml            = ET.SubElement(state_params_xml, DSN_XML_COEFFICIENTS)

    # ENCODING SETTINGS
    for key, val in usat_state_parameters.get(DSN_XML_SETTINGS, {}).items():
        encoding_settings_xml.set(key, str(val))

    for key, val in usat_state_parameters.get(DSN_XML_INPUT_AMBISONICS, {}).items():
        input_ambisonics_xml.set(key, str(val))

    for key, val in usat_state_parameters.get(DSN_XML_OUTPUT_AMBISONICS, {}).items():
        output_ambisonics_xml.set(key, str(val))

    input_layout_desc_xml.set(DSN_SMPL_INPUT_LAYOUT_DESC, usat_state_parameters[DSN_SMPL_INPUT_LAYOUT_DESC])
    output_layout_desc_xml.set(DSN_SMPL_OUTPUT_LAYOUT_DESC, usat_state_parameters[DSN_SMPL_OUTPUT_LAYOUT_DESC])

    # SPEAKER LAYOUTS
    pu.speaker_layout_to_xml(input_speaker_layout_xml, usat_state_parameters.get(DSN_XML_INPUT_SPEAKER_LAYOUT, []))
    pu.speaker_layout_to_xml(output_speaker_layout_xml, usat_state_parameters.get(DSN_XML_OUTPUT_SPEAKER_LAYOUT, []))

    # COEFFICIENTS
    for key, val in usat_state_parameters.get(DSN_XML_COEFFICIENTS, {}).items():
        coefficients_xml.set(key, str(val))

    return state_params_xml
    

def generate_decoding_data(args):

    yaml_file, seed = args
    if seed is not None:
        np.random.seed(seed)

    pretty_xml = None 

    print(f"Running in PID {os.getpid()} with seed {seed}")
    
    try:
        usat_state_parameters_dict  = parse_from_config(yaml_file)
        usat_state_parameters_xml   = build_xml_config(usat_state_parameters_dict)
        
        xml_bytes   = ET.tostring(usat_state_parameters_xml, encoding="utf-8", method="xml")
        parsed_xml  = minidom.parseString(xml_bytes)
        pretty_xml  = str(parsed_xml.toprettyxml(indent="  "))
        
        warnings.filterwarnings("ignore")
        output_dict = decode_for_random_parameter_generation(pretty_xml)
    
        return pretty_xml, output_dict
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error in PID {os.getpid()} with seed {seed}: {e}")
        print(f"Traceback:\n{tb_str}")
        output_dict = {
            "error_message": e,
            "traceback": tb_str 
        }
        return pretty_xml, output_dict


def main(num_decodings_targeted, yaml_path, bucket_name):
    start_time = time.time()

    if not yaml_path:
        print("No config file specified...")
        return

    # Create initial working directories
    dirs                = dir_utils.prepare_output_dir(yaml_path, bucket_name)
    config_base_name    = dirs[0] # Config name
    output_dir          = dirs[1] # Directory for this config file
    local_yaml_path     = dirs[2] # Path to yaml

    # Upload YAML to GC bucket if applicable 
    if bucket_name:
            yaml_blob = f"{config_base_name}/{config_base_name}.yaml"
            if not dir_utils.blob_exists(bucket_name, yaml_blob):
                dir_utils.upload_blob_to_gcs(local_file_path=local_yaml_path,
                                             bucket_name=bucket_name,
                                             destination_blob_name=yaml_blob)

    for i in range(1, num_decodings_targeted + 1):    
        print(f"Starting iteration {i}...")

        # Create directory for results
        seed        = secrets.randbits(32)
        results_dir = os.path.join(output_dir, f"seed_{seed}")

        # Generate data
        xml, output_dict = generate_decoding_data((local_yaml_path, seed))
        assert(isinstance(xml, str))

        # Save and serialise output data
        saved_dir = pu.save_output_data(xml, output_dict, seed, results_dir)
        print(f"Saved output files to: {saved_dir}")

        # Upload to GCS
        if bucket_name:
            print("Uploading to GCS...")
            gcs_dir = f"{config_base_name}/seed_{seed}"
            dir_utils.upload_directory_to_gcs(saved_dir, bucket_name, gcs_dir)

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random decoding data.")
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=10,
        help="Number of decodings to generate"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config (local or gs://...)"
    )
    parser.add_argument(
        "-b", "--bucket_name",
        type=str,
        default=None,
        help="GCS bucket to upload results to (optional)"
    )

    args = parser.parse_args()
    main(args.num, args.config, args.bucket_name)