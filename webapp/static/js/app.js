// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

// Event listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', removeImage);

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    // Hide error and result
    hideError();
    hideResult();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Upload and predict
    uploadAndPredict(file);
}

function uploadAndPredict(file) {
    const formData = new FormData();
    formData.append('image', file);

    showLoading();
    hideError();
    hideResult();

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        showResult(data);
    })
    .catch(err => {
        hideLoading();
        showError(err.error || 'Failed to predict ripeness. Please try again.');
    });
}

function showResult(data) {
    const ripenessLabel = document.getElementById('ripenessLabel');
    const confidence = document.getElementById('confidence');
    const probBars = document.getElementById('probBars');

    // Set ripeness label and confidence
    ripenessLabel.textContent = data.predicted_class;
    ripenessLabel.className = `ripeness-label ripeness-${data.predicted_class}`;
    confidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    // Create probability bars
    probBars.innerHTML = '';
    const sortedProbs = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([class_name, prob]) => {
        const probBar = document.createElement('div');
        probBar.className = 'prob-bar';

        const label = document.createElement('div');
        label.className = 'prob-label';
        label.textContent = class_name;

        const container = document.createElement('div');
        container.className = 'prob-bar-container';

        const fill = document.createElement('div');
        fill.className = 'prob-bar-fill';
        fill.style.width = `${prob * 100}%`;
        fill.textContent = prob >= 0.01 ? `${(prob * 100).toFixed(1)}%` : '';

        container.appendChild(fill);

        const value = document.createElement('div');
        value.className = 'prob-value';
        value.textContent = `${(prob * 100).toFixed(1)}%`;

        probBar.appendChild(label);
        probBar.appendChild(container);
        probBar.appendChild(value);

        probBars.appendChild(probBar);
    });

    resultSection.style.display = 'block';
}

function removeImage() {
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadArea.style.display = 'block';
    hideResult();
    hideError();
}

function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

function hideResult() {
    resultSection.style.display = 'none';
}

