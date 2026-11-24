/**
 * DrainSight AI - Front-end JavaScript Interactions
 * Manages drag & drop uploads, client-side validation,
 * upload progress visualization, and Chart.js dashboard metrics.
 */

document.addEventListener('DOMContentLoaded', function () {
  // -------------------------------------------------------------
  // Drag and Drop Zone Initialization
  // -------------------------------------------------------------
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const filePreview = document.getElementById('file-preview-card');
  const fileNameDisplay = document.getElementById('file-name-display');
  const fileSizeDisplay = document.getElementById('file-size-display');
  const uploadForm = document.getElementById('upload-form');
  const loadingOverlay = document.getElementById('loading-overlay');
  const uploadProgressBar = document.getElementById('upload-progress-bar');
  const uploadProgressContainer = document.getElementById('upload-progress-container');

  if (dropzone && fileInput) {
    // Prevent defaults
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // Highlight dropzone on drag
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleFileSelection(files[0]);
      }
    });

    // Handle normal file selection
    fileInput.addEventListener('change', function () {
      if (this.files.length > 0) {
        handleFileSelection(this.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    if (!file) return;

    // Check size limit (10MB for image, 50MB for video)
    const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|webm)$/i);
    const maxSizeBytes = isVideo ? 50 * 1024 * 1024 : 10 * 1024 * 1024;
    const maxMb = isVideo ? 50 : 10;

    if (file.size > maxSizeBytes) {
      alert(`File size exceeds the allowed limit of ${maxMb}MB.`);
      if (fileInput) fileInput.value = '';
      if (filePreview) filePreview.style.display = 'none';
      return;
    }

    if (filePreview && fileNameDisplay && fileSizeDisplay) {
      fileNameDisplay.textContent = file.name;
      fileSizeDisplay.textContent = formatBytes(file.size);
      filePreview.style.display = 'block';
    }
  }

  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  // Handle Form Submission & Animated Progress
  if (uploadForm) {
    uploadForm.addEventListener('submit', function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please choose a file to upload first.');
        return;
      }

      if (uploadProgressContainer && uploadProgressBar) {
        uploadProgressContainer.style.display = 'block';
        let progress = 10;
        uploadProgressBar.style.width = progress + '%';
        uploadProgressBar.textContent = progress + '%';

        const timer = setInterval(() => {
          if (progress < 90) {
            progress += 10;
            uploadProgressBar.style.width = progress + '%';
            uploadProgressBar.textContent = progress + '%';
          } else {
            clearInterval(timer);
          }
        }, 200);
      }

      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
      }
    });
  }

  // -------------------------------------------------------------
  // Dashboard Chart.js Integration
  // -------------------------------------------------------------
  const statusChartCanvas = document.getElementById('statusChart');
  const wasteChartCanvas = document.getElementById('wasteChart');

  if (statusChartCanvas && typeof Chart !== 'undefined') {
    const clearCnt = parseInt(statusChartCanvas.dataset.clear || '0', 10);
    const partialCnt = parseInt(statusChartCanvas.dataset.partial || '0', 10);
    const heavyCnt = parseInt(statusChartCanvas.dataset.heavy || '0', 10);

    new Chart(statusChartCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Clear (0-20%)', 'Partially Blocked (21-60%)', 'Heavily Blocked (>60%)'],
        datasets: [{
          data: [clearCnt, partialCnt, heavyCnt],
          backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' }
        },
        cutout: '65%'
      }
    });
  }

  if (wasteChartCanvas && typeof Chart !== 'undefined') {
    const plastic = parseInt(wasteChartCanvas.dataset.plastic || '0', 10);
    const leaves = parseInt(wasteChartCanvas.dataset.leaves || '0', 10);
    const paper = parseInt(wasteChartCanvas.dataset.paper || '0', 10);
    const mud = parseInt(wasteChartCanvas.dataset.mud || '0', 10);

    new Chart(wasteChartCanvas, {
      type: 'bar',
      data: {
        labels: ['Plastic Waste', 'Leaves / Plant', 'Paper / Cardboard', 'Mud / Sediment'],
        datasets: [{
          label: 'Detections Count',
          data: [plastic, leaves, paper, mud],
          backgroundColor: ['#ef4444', '#10b981', '#f59e0b', '#8b5cf6'],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { precision: 0 }
          }
        }
      }
    });
  }
});
