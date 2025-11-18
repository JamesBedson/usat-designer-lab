import xml.etree.ElementTree as ET
from usat_designer.processing.constants import *
from usat_designer.processing.optimize_usat_designer import optimize_for_usat_designer

def decode_for_random_parameter_generation(xml_string: str) -> dict:
    usat_state_parameters_xml   = ET.fromstring(xml_string)
    optimization_dict           = parse_encoding_settings(usat_state_parameters_xml)
    
    optimization_dict["show_results"]       = False
    optimization_dict["save_results"]       = False
    optimization_dict["results_file_name"]  = None
    
    output_data = optimize_for_usat_designer(optimization_dict) 
    print(output_data.keys())
    return output_data