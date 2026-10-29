// Get important HTML elements
const imageInput = document.getElementById("foodImage");
const imagePreview = document.getElementById("imagePreview");


// Show image preview when user selects an image
imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {

        const imageURL = URL.createObjectURL(file);

        imagePreview.src = imageURL;

        imagePreview.style.display = "block";
    }
});


// Analyze food
async function analyzeFood() {

    const result = document.getElementById("result");

    if (imageInput.files.length === 0) {

        result.innerHTML =
            "<p>Please select a food image first.</p>";

        return;
    }


    const image = imageInput.files[0];


    // Create form data
    const formData = new FormData();

    formData.append("image", image);


    // Show loading message
    result.innerHTML =
        "<p>🤖 Analyzing food... Please wait.</p>";


    try {

        // Send image to Flask backend
        const response = await fetch(
            "http://127.0.0.1:5000/analyze",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        // Check backend error
        if (data.error) {

            result.innerHTML =
                `<p>${data.error}</p>`;

            return;
        }


        // Display food result
        result.innerHTML = `
            <h3>Food Detected 🍛</h3>

            <p>
                <strong>Food:</strong>
                ${data.food}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${data.confidence}%
            </p>
        `;


        // Display nutrition if available
        if (data.nutrition) {

            const nutrition = data.nutrition;


            document.getElementById("calories").textContent =
                `${nutrition.calories} kcal`;

            document.getElementById("carbohydrates").textContent =
                `${nutrition.carbohydrates} g`;

            document.getElementById("protein").textContent =
                `${nutrition.protein} g`;

            document.getElementById("fats").textContent =
                `${nutrition.fats} g`;

            document.getElementById("free-sugar").textContent =
                `${nutrition.free_sugar} g`;

            document.getElementById("fibre").textContent =
                `${nutrition.fibre} g`;

            document.getElementById("sodium").textContent =
                `${nutrition.sodium} mg`;

            document.getElementById("calcium").textContent =
                `${nutrition.calcium} mg`;

            document.getElementById("iron").textContent =
                `${nutrition.iron} mg`;

            document.getElementById("vitamin-c").textContent =
                `${nutrition.vitamin_c} mg`;

            document.getElementById("folate").textContent =
                `${nutrition.folate} µg`;
        }


    } catch (error) {

        result.innerHTML = `
            <p>
                ❌ Could not connect to Python backend.
            </p>
        `;

        console.error(error);
    }
}