// The API can be hosted elsewhere by setting window.DIETAI_API_URL before this file loads.
const API_BASE_URL = (window.DIETAI_API_URL || "http://127.0.0.1:8000")
    .replace(/\/+$/, "");

const imageInput = document.getElementById("foodImage");
const imagePreview = document.getElementById("imagePreview");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (!file) {
        imagePreview.removeAttribute("src");
        imagePreview.style.display = "none";
        return;
    }

    if (imagePreview.dataset.objectUrl) {
        URL.revokeObjectURL(imagePreview.dataset.objectUrl);
    }

    const imageURL = URL.createObjectURL(file);
    imagePreview.dataset.objectUrl = imageURL;
    imagePreview.src = imageURL;
    imagePreview.style.display = "block";
});

function showMessage(message) {
    // Treat server-provided text as text, never as HTML.
    result.textContent = message;
}

async function analyzeFood() {
    if (imageInput.files.length === 0) {
        showMessage("Please select a food image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);
    showMessage("Uploading image for validation...");

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            body: formData
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            showMessage(data.detail || data.error || `Request failed (HTTP ${response.status}).`);
            return;
        }

        // The current backend scaffold does not return predictions.
        if (!data.food || !data.nutrition) {
            showMessage("The API responded, but no food or nutrition prediction was provided.");
            return;
        }

        showMessage(`Food detected: ${data.food}`);

        const nutrition = data.nutrition;
        const fields = {
            calories: [nutrition.calories, "kcal"],
            carbohydrates: [nutrition.carbohydrates, "g"],
            protein: [nutrition.protein, "g"],
            fats: [nutrition.fats, "g"],
            "free-sugar": [nutrition.free_sugar, "g"],
            fibre: [nutrition.fibre, "g"],
            sodium: [nutrition.sodium, "mg"],
            calcium: [nutrition.calcium, "mg"],
            iron: [nutrition.iron, "mg"],
            "vitamin-c": [nutrition.vitamin_c, "mg"],
            folate: [nutrition.folate, "µg"]
        };

        for (const [id, [value, unit]] of Object.entries(fields)) {
            if (value !== undefined && value !== null) {
                document.getElementById(id).textContent = `${value} ${unit}`;
            }
        }
    } catch (error) {
        showMessage("Could not connect to the DietAI API. Check that the backend is running.");
        console.error(error);
    }
}
