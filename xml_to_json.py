import xmltodict
import json


def convert_xml_to_json(xml_input_path, json_output_path):
    """
    Converts an XML file to a JSON file.

    :param xml_input_path: Path to the input XML file.
    :param json_output_path: Path where the output JSON file will be saved.
    """
    try:
        with open(xml_input_path, "r", encoding="utf-8") as xml_file:
            # Parse the XML file into an ordered dictionary
            xml_dict = xmltodict.parse(xml_file.read())

        # Convert the ordered dictionary to a regular dictionary (optional)
        # xml_dict = json.loads(json.dumps(xml_dict))

        with open(json_output_path, "w", encoding="utf-8") as json_file:
            # Convert the dictionary to a JSON string with indentation for readability
            json.dump(xml_dict, json_file, indent=4, ensure_ascii=False)

        print(f"Successfully converted '{xml_input_path}' to '{json_output_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    xml_input = "wikivoyage.xml"  # Path to your input XML file
    json_output = "wikivoyage.json"  # Desired path for the output JSON file

    convert_xml_to_json(xml_input, json_output)
