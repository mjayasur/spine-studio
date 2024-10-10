// Opens the modal
function openPopup() {
    document.getElementById("uploadModal").style.display = "block";
}

// Closes the modal
function closePopup() {
    document.getElementById("uploadModal").style.display = "none";
}

function handleModelChange() {
    const imagingType = document.getElementById("type-of-imaging").value;
    const fileUploadBox = document.getElementById('fileUploadBox');
    const textInputBox = document.getElementById('textInputBox');
    const actionButton = document.getElementById('actionButton');
    const insuranceFields = document.getElementById('insuranceFields');

    if (imagingType === "insurance") {
        // Hide file upload and show the radiology report input and insurance fields
        fileUploadBox.style.display = 'none';
        textInputBox.style.display = 'block';
        insuranceFields.style.display = 'block';
        actionButton.textContent = 'Calculate Probability';
    } else {
        // Show file upload and hide the radiology report input and insurance fields
        fileUploadBox.style.display = 'block';
        textInputBox.style.display = 'none';
        insuranceFields.style.display = 'none';
        actionButton.textContent = 'Upload Files';
    }
}
// Handles the submission when the button is clicked, either upload files or calculate probability
function handleUploadClick() {
    const imagingType = document.getElementById("type-of-imaging").value;

    // Check if the selected option is for Insurance calculation
    if (imagingType === "insurance") {
        const radiologyReport = document.getElementById('radiologyReport').value;

        if (!radiologyReport.trim()) {
            alert('Please enter your radiology report.');
            return;
        }

        // Encode the radiology report for use in the URL
        const encodedReport = encodeURIComponent(radiologyReport);

        // Perform a GET request to the server-side endpoint
        const url = `/calculate-insurance?radiology-report=${encodedReport}`;
        fetch(url, {
            method: 'GET'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();  // Parse the JSON from the response
            })
            .then(data => {
                // Display the results in the 'results' div
                const resultsDiv = document.getElementById('results');

                // Construct the HTML for displaying the results
                resultsDiv.innerHTML = `
                <h3>Results:</h3>
                <p><strong>Feature Detection:</strong> ${data["Feature.detection"]}</p>
                <p><strong>Probability of Insurance Coverage:</strong> ${data["Probability.of.insurance.coverage"]}</p>
            `;
            })
            .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
            });

    } else if (imagingType === "CT" || imagingType === "MRI" || imagingType === "XR") {
        // Proceed with the file upload for CT, MRI, or XR
        console.log('Redirecting to loading page'); // Debugging line
        document.getElementById("loading-frame").hidden = false;
        document.getElementById("index-content").hidden = true;

        var formData = new FormData(document.getElementById('uploadForm'));

        fetch('/upload-file', {
            method: 'POST',
            body: formData
        }).then(response => {
            if (response.ok) {
                response.json().then(data => {
                    var redirectUrl;
                    if (imagingType === "CT") {
                        console.log(data);
                        redirectUrl = `/output_ct?id=${data['uuid']}&modality=CT&availableLevels=${data['vertebrae']}`;
                    } else if (imagingType === "MRI") {
                        redirectUrl = "/output_mri";
                    } else if (imagingType === "XR") {
                        redirectUrl = `/output_xr?id=${data['uuid']}`;
                    }
                    window.location.href = redirectUrl;
                });
            } else {
                alert('Failed to upload files.');
            }
        }).catch(error => {
            console.error('Error:', error);
            alert('Error uploading files.');
        }).finally(() => {
            document.getElementById("loading-frame").hidden = true;
            document.getElementById("index-content").hidden = false;
        });
    } else {
        alert('Please select a valid imaging type to proceed.');
    }
}

// Close the modal when clicking outside of the modal content
window.onclick = function (event) {
    var modal = document.getElementById("uploadModal");
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

// Function to handle the insurance calculation
function calculateInsuranceProbability(radiologyReport) {
    console.log('Calculating insurance probability with radiology report:', radiologyReport);
    window.location.href = "/output_insurance?report=" + radiologyReport;

}
