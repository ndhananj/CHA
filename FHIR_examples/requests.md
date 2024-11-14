
# Search patients, Response is in Get_Patient.json
curl -X GET "https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/Patient" \
-H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
-H "Accept: application/fhir+json"

# Search patients by name, Response is in Get_Patient_Smith.json
curl -X GET "https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/Patient?name=Smith" \
-H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
-H "Accept: application/fhir+json"

# Get a specific patient, Response is in Get_Patient_Specific.json
curl -X GET "https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/Patient/7e87e2f6-e579-499d-abc7-fb7447d326be" \
-H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
-H "Accept: application/fhir+json"

# Get a specific patient's medications, Response is in Get_Patient_Specific_Medications.json
curl -X GET "https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/MedicationRequest?patient=7e87e2f6-e579-499d-abc7-fb7447d326be" \
-H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
-H "Accept: application/fhir+json"

# Get  a specific patient's procedures, Response is in Get_Patient_Specific_Procedures.json
curl -X GET "https://fhir.careevolution.com/Master.Adapter1.WebClient/api/fhir-r4/Procedure?patient=7e87e2f6-e579-499d-abc7-fb7447d326be" \
-H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
-H "Accept: application/fhir+json"
