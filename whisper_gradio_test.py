import gradio as gr
import time
import numpy as np
from transformers import pipeline
import requests
import pandas as pd
import os
from openai import OpenAI
from typing import Dict, List, Any, Optional
from functools import reduce

# Define field mappings for MedicationRequest
medication_field_mappings = {
    "Medication": ["medicationCodeableConcept", "coding", 0, "display"],
    "Dosage Instructions": ["dosageInstruction", 0, "text"],
    "Prescribed Date": ["authoredOn"]
}

# Define field mappings for Procedure
procedure_field_mappings = {
    "Procedure": ["code", "coding", 0, "display"],
    "Status": ["status"],
    "Performed Date": ["performedDateTime"]
}

# Define field mappings for Allergies
allergy_mappings = {
    "Allergen": ["code", "text"],
    "Reaction": ["reaction", 0, "description"],
    "Severity": ["criticality"],
    "Status": ["clinicalStatus", "coding", 0, "code"],
    "Onset Date": ["onsetDateTime"],
    "Recorded Date": ["recordedDate"],
    "Category": ["category", 0],
    "Manifestation": ["reaction", 0, "manifestation", 0, "text"]
}

# Define field mappings for Conditions
condition_mappings = {
    "Category": ["category", 0, "coding", 0, "display"],
    "Onset Date": ["onsetDateTime"],
    "ICD10": ["code", "coding", 2, "code"],  # Index 2 typically has ICD-10 coding
    "SNOMED": ["code", "coding", 0, "code"]  # Index 0 typically has SNOMED coding
}

endpoint_to_fieldmapping = {
    "MedicationRequest":medication_field_mappings, 
    "Procedure":procedure_field_mappings,
    "AllergyIntolerance":allergy_mappings,
    "Condition":condition_mappings
}

def get_nested_value(data: Any, path: List[str], default: Any = None) -> Any:
    """
    Safely retrieves a nested value from a dictionary using a path list.

    Args:
        data: The data structure to traverse (dict, list, or other)
        path: List of keys/indices to follow
        default: Value to return if path is not found

    Returns:
        The value at the specified path or the default value
    """
    try:
        return reduce(lambda d, key: d[key] if isinstance(d, (dict, list)) and d else default,
                     path,
                     data)
    except (KeyError, IndexError, TypeError):
        return default

def safe_to_datetime(value: Any) -> Optional[pd.Timestamp]:
    """
    Safely convert a value to datetime, returning None if conversion fails.

    Args:
        value: Value to convert to datetime

    Returns:
        Pandas Timestamp if conversion successful, None otherwise
    """
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return None

def extract_info(row: pd.Series, field_mappings: Dict[str, List[str]]) -> pd.Series:
    """
    Extracts specified fields from a single row of the DataFrame based on field mappings.

    Args:
        row: A pandas Series containing data for a FHIR resource
        field_mappings: A dictionary that maps user-friendly field names to paths in the data structure

    Returns:
        pd.Series with extracted information based on field mappings
    """
    if None==field_mappings: # no mappings exist
        return row

    extracted_data = {}

    for field_name, path in field_mappings.items():
        # Convert row to dictionary if it's a Series
        data = row.to_dict() if isinstance(row, pd.Series) else row

        # Get the nested value using the safe accessor
        value = get_nested_value(data, path)

        # Handle different types of values
        if value is None:
            extracted_data[field_name] = None
        elif isinstance(value, (str, int, float, bool)):
            extracted_data[field_name] = value
        elif isinstance(value, (dict, list)):
            # Convert complex objects to string representation
            extracted_data[field_name] = str(value)
        else:
            extracted_data[field_name] = None

    return pd.Series(extracted_data)

def process_fhir_data(endpoint:str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a FHIR DataFrame into a clean summary DataFrame based on specified field mappings.
    
    Args:
        endpoint:  needed to determine correct field mappings
        df: DataFrame containing FHIR resource data

    Returns:
        DataFrame with processed FHIR resource information
    """
    print("df dimensions", df.shape)

     # determine field_mappings from endpoint 
     #field_mappings: A dictionary that maps user-friendly field names to paths in the data structure
    field_mappings = endpoint_to_fieldmapping.get(endpoint,None)
    print(field_mappings)

    try:
        result_df = df.apply(lambda row: extract_info(row, field_mappings), axis=1)

        # Convert datetime strings to pandas datetime objects where possible
        date_columns = [col for col in result_df.columns if any(date_term in col.lower()
                       for date_term in ['date', 'time', 'authored'])]

        for col in date_columns:
            result_df[col] = result_df[col].apply(safe_to_datetime)

        return result_df

    except Exception as e:
        raise ValueError(f"Error processing FHIR data: {str(e)}")



def get_fhir_endpoint(query):
    """
    Determines the appropriate FHIR endpoint based on the query string using an LLM.
    
    Args:
        query (str): The search query string
        
    Returns:
        str: The matching FHIR endpoint or None if no match is found
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ['NVIDIA_API_KEY']
    )

    sys_prompt = """
    You are tasked with determining the correct endpoint for a healthcare data request based on a set of predefined endpoints. Given a natural language request, return the single most relevant endpoint from the following list **exactly as shown**, in the **same capitalization and format**:

    1. `AllergyIntolerance`
    2. `CarePlan`
    3. `CareTeam`
    4. `Condition`
    5. `Device`
    6. `DiagnosticReport`
    7. `DocumentReference`
    8. `Encounter`
    9. `Goal`
    10. `Immunization`
    11. `Location`
    12. `MedicationRequest`
    13. `Observation`
    14. `Organization`
    15. `Patient`
    16. `Practitioner`
    17. `Procedure`
    18. `ServiceRequest`

    Return only one endpoint from this list per request, and match it exactly to one of these predefined terms. Do not include any explanations, extra text, or punctuation, and do not deviate from these terms. 

    For example:

    - If the request is "Show me my medications," return: `MedicationRequest`
    - If the request is "I need to see my allergies," return: `AllergyIntolerance`

    If the request does not clearly match any of the endpoints, make the best-guess match based on context but still return only one endpoint exactly as listed above.
    """

    completion = client.chat.completions.create(
        model="meta/llama3-8b-instruct",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0,  # Set to 0 for deterministic output
        top_p=1,
        max_tokens=16,
        stream=True
    )

    # Collect the streamed response
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    # Clean up the response (remove any backticks if present)
    endpoint = response.strip('` \n')
    
    return endpoint if endpoint else None



def process_health_request(endpoint) -> pd.DataFrame:

    who_has_what = {
               'AllergyIntolerance': '4f301ddd-1a7f-4c9b-883e-1db9c5c7511d', 
               'Condition': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
               'DiagnosticReport': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
               'Encounter': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
               'Immunization': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
               'MedicationRequest': 'd1fb6ccf-30c9-4825-9c74-824c36e86fbc', 
               'Observation': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
               'Procedure': 'd1fb6ccf-30c9-4825-9c74-824c36e86fbc'
               }

    access_token = os.environ['ACCESS_TOKEN']
    patient_id   = who_has_what.get(endpoint, None)

    if endpoint:
        url = f"https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/{endpoint}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json"
        }
        params = {"patient": patient_id}
        
        print(f"curl -X GET '{requests.Request('GET', url, headers=headers, params=params).prepare().url}' " + " ".join(f"-H '{k}: {v}'" for k,v in headers.items()))
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        # Convert FHIR Bundle to dataframe
        if 'entry' in data:
            records = [item['resource'] for item in data['entry']]
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()  # Empty dataframe if no results
    
    return pd.DataFrame()  # Empty dataframe if request not understood


def sequential_process(audio):
    if audio is None:
        return ["No audio recorded", "", "", None]
    
    # First pane shows audio recording status and other 4 blank 
    status = "Audio recorded successfully"
    yield [status, "", "", None]
    
    # Second pane: Whisper transcription
    time.sleep(0.5)  # Give UI time to update
    sr, y = audio
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    transcribed_text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    #  update transcription, and set others to same
    yield [status, transcribed_text, "", None]

    # Third pane: Generate endpoint
    time.sleep(1)
    endpoint = get_fhir_endpoint(transcribed_text)
    yield [status, transcribed_text, endpoint, None]
    
    # output 5: Make API call and show dataframe
    time.sleep(1)

    df = process_health_request(endpoint)
    medication_summary_df = process_fhir_data(endpoint, df)

    yield [status, transcribed_text, endpoint, medication_summary_df]

def reset_audio():
    return None

# Sample test prompts for audio input
test_prompts = """
    Show me my medications
    I need to see my allergies 
    'AllergyIntolerance': '4f301ddd-1a7f-4c9b-883e-1db9c5c7511d', 
    'Condition': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
    'DiagnosticReport': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
    'Encounter': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
    'Immunization': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
    'MedicationRequest': 'd1fb6ccf-30c9-4825-9c74-824c36e86fbc', 
    'Observation': '2b8f2802-9a24-44a6-847b-0c5672f80824', 
    'Procedure': 'd1fb6ccf-30c9-4825-9c74-824c36e86fbc'
"""

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Audio Processing Pipeline")
    
    with gr.Row():
        # Left column for audio input and controls
        with gr.Column(scale=2):
            # Input audio component
            input_audio = gr.Audio(
                sources="microphone",
                type="numpy",
                label="Record Audio"
            )
            
            with gr.Row():
                # Button to start processing
                process_btn = gr.Button("Process Recording")
                # Reset button
                reset_btn = gr.Button("Reset Audio")
            
            # Output panes
            output1 = gr.Textbox(label="Recording Status")
            output2 = gr.Textbox(label="Query")
            output3 = gr.Textbox(label="Endpoint")
            output4 = gr.Dataframe(label="DataFrame")
        
        # Right column for test prompts
        with gr.Column(scale=1):
            gr.Markdown("### Sample Test Prompts")
            for prompt in test_prompts.splitlines():
                gr.Markdown(f"- {prompt}")
    
    # Connect the buttons to their functions
    process_btn.click(
        fn=sequential_process,
        inputs=[input_audio],
        outputs=[output1, output2, output3, output4]
    )
    
    reset_btn.click(
        fn=reset_audio,
        inputs=[],
        outputs=[input_audio]
    )

# Launch the interface
demo.queue().launch()