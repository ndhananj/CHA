from openai import OpenAI
from google.colab import userdata

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
        api_key=userdata.get('NVIDIA_API_KEY')
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

def process_health_request(query, patient_id, access_token):
    import requests
    import pandas as pd
    
    endpoint = get_fhir_endpoint(query)
    
    if endpoint:
        url = f"https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/{endpoint}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json"
        }
        params = {"patient": patient_id}
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        # Convert FHIR Bundle to dataframe
        if 'entry' in data:
            records = [item['resource'] for item in data['entry']]
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()  # Empty dataframe if no results
    
    return pd.DataFrame()  # Empty dataframe if request not understood
