from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from usat_designer.processing.launch_usat import parse_coefficients, create_speaker_layout
from usat_designer.processing.constants import *

import numpy as np
import os
import json
import xml.etree.ElementTree as ET

def speaker_layout_to_xml(parent: ET.Element, layout_list: list) -> ET.Element:
    
    for idx, speaker in enumerate(layout_list, start = 1):
        tag_name        = f"Speaker_{idx}"
        speaker_elem    = ET.SubElement(parent, tag_name)
        
        for k, v in speaker.items():
            speaker_elem.set(k, str(v))
    
    return parent

def serialize_coordinates(obj):
    if isinstance(obj, MyCoordinates):
        return {
            "_type": "MyCoordinates",
            "spherical_deg": obj.sph_deg().tolist()
        }
    elif isinstance(obj, list):
        return [serialize_coordinates(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: serialize_coordinates(v) for k, v in obj.items()}
    else:
        return obj
    
def restore_coordinates(serialized):
    if isinstance(serialized, dict) and serialized.get("_type") == "MyCoordinates":
        sph_deg = np.array(serialized["spherical_deg"])
        return MyCoordinates.mult_points(sph_deg)
    elif isinstance(serialized, list):
        return [restore_coordinates(v) for v in serialized]
    elif isinstance(serialized, dict):
        return {k: restore_coordinates(v) for k, v in serialized.items()}
    else:
        return serialized
    
def save_output_data(xml_string: str,
                    output_dict: dict,
                    seed: float,
                    output_dir: str):

    os.makedirs(output_dir, exist_ok=True)

    xml_path = os.path.join(output_dir, f"y_parameters_{seed}.xml")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

    if "error_message" in output_dict:
        error_path = os.path.join(output_dir, "error.txt")
        
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(output_dict["error_message"])
            f.write(output_dict["traceback"])

    else:
        matrix_path = os.path.join(output_dir, f"matrix_data_{seed}.npz")
        
        np.savez(matrix_path,
        **{
            DSN_OUT_SPEAKER_MATRIX: output_dict[DSN_OUT_SPEAKER_MATRIX],
            DSN_OUT_ENCODING_MATRIX: output_dict[DSN_OUT_ENCODING_MATRIX],
            DSN_OUT_TRANSCODING_MATRIX: output_dict[DSN_OUT_TRANSCODING_MATRIX],
            DSN_OUT_DECODING_MATRIX: output_dict[DSN_OUT_DECODING_MATRIX],
        }
)
        
        metadata = {
            DSN_SMPL_SEED: seed,
            DSN_OUT_OUTPUT_LAYOUT: serialize_coordinates(output_dict[DSN_OUT_OUTPUT_LAYOUT]),
            DSN_OUT_CLOUD: serialize_coordinates(output_dict[DSN_OUT_CLOUD]),
        }

        metadata_path = os.path.join(output_dir, f"metadata_{seed}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir

import xml.etree.ElementTree as ET

def create_speaker_dict(speaker_layout):
    speakers = []
    if speaker_layout is not None:
        for speaker in speaker_layout:
            speakers.append({
                DSN_SPK_CHANNEL_ID: int(speaker.attrib.get(DSN_SPK_CHANNEL_ID, 0)),
                DSN_SPK_AZIMUTH: float(speaker.attrib.get(DSN_SPK_AZIMUTH, 0.0)),
                DSN_SPK_ELEVATION: float(speaker.attrib.get(DSN_SPK_ELEVATION, 0.0)),
                DSN_SPK_DISTANCE: float(speaker.attrib.get(DSN_SPK_DISTANCE, 1.0)),
            })


def usat_xml_to_dict(xml_string: str):
    
    root = ET.fromstring(xml_string)
    data = {}

    encoding_settings       = root.find(DSN_XML_SETTINGS)
    data[DSN_XML_SETTINGS]  = encoding_settings.attrib if encoding_settings is not None else {}

    input_ambisonics                = root.find(DSN_XML_INPUT_AMBISONICS)
    data[DSN_XML_INPUT_AMBISONICS]  = input_ambisonics.attrib if input_ambisonics is not None else {}

    output_ambisonics               = root.find(DSN_XML_OUTPUT_AMBISONICS)
    data[DSN_XML_OUTPUT_AMBISONICS] = output_ambisonics.attrib if output_ambisonics is not None else {}

    input_layout_desc                   = root.find(DSN_SMPL_INPUT_LAYOUT_DESC)
    data[DSN_SMPL_INPUT_LAYOUT_DESC]    = input_layout_desc.attrib if input_layout_desc is not None else {}

    output_layout_desc                  = root.find(DSN_SMPL_OUTPUT_LAYOUT_DESC)
    data[DSN_SMPL_OUTPUT_LAYOUT_DESC]   = output_layout_desc.attrib if output_layout_desc is not None else {}

    data[DSN_XML_INPUT_SPEAKER_LAYOUT]  = []
    data[DSN_XML_OUTPUT_SPEAKER_LAYOUT] = []
    
    if data[DSN_XML_SETTINGS][DSN_XML_INPUT_TYPE] == DSN_XML_SPEAKER_LAYOUT:
        input_speaker_layout = root.find(DSN_XML_INPUT_SPEAKER_LAYOUT)
        assert(input_speaker_layout is not None)
        data[DSN_XML_INPUT_SPEAKER_LAYOUT] = create_speaker_layout(input_speaker_layout)

    if data[DSN_XML_SETTINGS][DSN_XML_OUTPUT_TYPE] == DSN_XML_SPEAKER_LAYOUT:
        output_speaker_layout = root.find(DSN_XML_OUTPUT_SPEAKER_LAYOUT)
        assert(output_speaker_layout is not None)
        data[DSN_XML_OUTPUT_SPEAKER_LAYOUT] = create_speaker_layout(output_speaker_layout) 
    
    coefficients_xml = root.find(DSN_XML_COEFFICIENTS)
    assert(coefficients_xml is not None)
    data[DSN_XML_COEFFICIENTS] = parse_coefficients(coefficients_xml)

    return data
