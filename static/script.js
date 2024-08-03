function openPopup() {
    document.getElementById("uploadModal").style.display = "block";
}

function closePopup() {
    document.getElementById("uploadModal").style.display = "none";
}

function handleUploadClick() {
    var imagingType = document.getElementById("type-of-imaging").value;
    console.log('Imaging Type Selected:', imagingType); // Debugging line
    if (imagingType === "CT" || imagingType === "MRI" || imagingType === "XR") {
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
                        redirectUrl = "/output_ct";
                    } else if (imagingType === "MRI") {
                        redirectUrl = "/output_mri";
                    } else if (imagingType === "XR") {
                        redirectUrl = "/output_xr";
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
