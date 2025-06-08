// static/style.js

document.addEventListener('DOMContentLoaded', function() {
    // Get the form and the processing overlay
    const uploadForm = document.getElementById('uploadForm');
    const processingOverlay = document.getElementById('processing-overlay');
    const processingSteps = document.getElementById('processing-steps').getElementsByTagName('li');

    if (uploadForm && processingOverlay) {
        // Listen for the form submission
        uploadForm.addEventListener('submit', function(event) {
            // Prevent the default form submission (which would just navigate away)
            // event.preventDefault(); // We won't prevent default here, as we want the form to submit normally

            // Show the processing overlay immediately
            processingOverlay.style.display = 'flex'; // Use flex to apply center alignment styles

            // Note: Without more advanced techniques (like AJAX, SSE, WebSockets),
            // we cannot get real-time updates from the synchronous Flask backend.
            // The steps below are just for visual representation on the client side
            // right after submission. True step highlighting would need server communication.

            // Optional: You could try to visually indicate the start of steps here,
            // but the browser will block until the new page is loaded.
            // For example: processingSteps[0].classList.add('active');

            // Allow the form to submit after showing the overlay.
            // The page will reload with the results or an error when the server responds.
            // The overlay will disappear when the new page loads.

            // If you *did* preventDefault, you would then use fetch() to send the form data
            // and handle the response and updates asynchronously, which is more complex.
            // For this simple approach, just showing the overlay is enough feedback.
        });

        // Optional: Hide the overlay if the page loads and there's an error message
        // (Means the /process route returned an error instead of redirecting to index without errors)
        const errorMessage = document.querySelector('p[style="color: red;"]');
        if (errorMessage) {
             processingOverlay.style.display = 'none';
        }

         // Optional: Hide the overlay if results are displayed on load (e.g., after a successful process)
         const resultsContainer = document.querySelector('.results-container');
         if (resultsContainer) {
             processingOverlay.style.display = 'none';
         }

    } else {
        console.error("Upload form or processing overlay not found!");
    }
});

// Note: To implement actual step-by-step updates on the steps list,
// you would need to use AJAX to send the form data,
// have the Flask backend run the processing in a separate thread/process,
// store the processing status server-side (e.g., in a dictionary by unique ID),
// and have the client-side JavaScript periodically poll a Flask /status/<id> endpoint
// to fetch the current step and update the list visuals. This is significantly more complex.