// Global variables for storing aspect ratios
let canvasAspectRatio = 1;
let answerAspectRatio = 1;
let isFullscreen = false;
let toolPosition = 'bottom'; // Default position

// Settings functions - FIXED
function openSettings() {
  document.getElementById('settingsOverlay').style.display = 'block';
  document.getElementById('settingsPanel').style.display = 'block';
  document.getElementById('toolPositionSelect').value = toolPosition;
}

function closeSettings() {
  document.getElementById('settingsOverlay').style.display = 'none';
  document.getElementById('settingsPanel').style.display = 'none';
}

// FIXED: Tool position function
function updateToolPosition(position) {
  const controls = document.getElementById('editorControls');
  const canvasContainer = document.getElementById('canvasContainer');
  const combinedSection = document.getElementById('combinedSection');
  
  toolPosition = position;
  
  if (position === 'top') {
    controls.className = 'top-position';
    // Move controls before canvas container
    combinedSection.insertBefore(controls, canvasContainer);
  } else {
    controls.className = 'bottom-position';
    // Move controls after canvas container
    if (canvasContainer.nextSibling) {
      combinedSection.insertBefore(controls, canvasContainer.nextSibling);
    } else {
      combinedSection.appendChild(controls);
    }
  }
}

// Function to calculate optimal canvas size based on available space
function calculateOptimalCanvasSize(imageWidth, imageHeight, forceFullArea = false) {
  let maxWidth, maxHeight;
  
  if (forceFullArea || isFullscreen) {
    // Use most of the viewport, accounting for controls
    maxWidth = window.innerWidth - 40;
    maxHeight = window.innerHeight - 180; // Account for controls and margins
  } else {
    // Use generous portion of viewport for better viewing
    maxWidth = Math.min(window.innerWidth - 40, window.innerWidth * 0.95);
    maxHeight = Math.min(window.innerHeight - 250, window.innerHeight * 0.7);
  }
  
  const imageAspectRatio = imageHeight / imageWidth;
  
  let optimalWidth = maxWidth;
  let optimalHeight = optimalWidth * imageAspectRatio;
  
  // If height exceeds max height, scale based on height instead
  if (optimalHeight > maxHeight) {
    optimalHeight = maxHeight;
    optimalWidth = optimalHeight / imageAspectRatio;
  }
  
  // Ensure minimum sizes
  optimalWidth = Math.max(optimalWidth, 400);
  optimalHeight = Math.max(optimalHeight, 300);
  
  return {
    width: Math.floor(optimalWidth),
    height: Math.floor(optimalHeight)
  };
}

// Function to resize canvas to fit viewing area
function resizeCanvasToFit() {
  const container = document.getElementById("canvasContainer");
  const bgCanvas = document.getElementById("bgCanvas");
  
  if (bgCanvas.width > 0 && bgCanvas.height > 0) {
    const optimalSize = calculateOptimalCanvasSize(bgCanvas.width, bgCanvas.height, true);
    container.style.width = optimalSize.width + "px";
    container.style.height = optimalSize.height + "px";
  }
}

// Fullscreen functionality - FIXED
function toggleFullscreen() {
  const container = document.getElementById("canvasContainer");
  const btn = document.getElementById("fullscreenBtn");
  
  if (!isFullscreen) {
    container.classList.add("fullscreen-mode");
    btn.textContent = "Exit Fullscreen";
    isFullscreen = true;
    
    // Resize to fill screen
    container.style.width = "100vw";
    container.style.height = "100vh";
  } else {
    container.classList.remove("fullscreen-mode");
    btn.textContent = "Fullscreen";
    isFullscreen = false;
    
    // Restore normal size
    resizeCanvasToFit();
  }
}

// ESC key to exit fullscreen
document.addEventListener("keydown", function(e) {
  if (e.key === "Escape" && isFullscreen) {
    toggleFullscreen();
  }
  if (e.key === "Escape" && document.getElementById('settingsPanel').style.display === 'block') {
    closeSettings();
  }
});

// Function to process ZIP file (whether from file upload or URL)
function processZipFile(zipData) {
  return JSZip.loadAsync(zipData).then(function(zip) {
    let combinedEntry = null, answerEntry = null, textEntry = null;
    zip.forEach(function(relativePath, zipEntry) {
      if (!zipEntry.dir) {
        const lowerName = relativePath.toLowerCase();
        if (lowerName.includes("combined") && !combinedEntry) {
          combinedEntry = zipEntry;
        }
        if (lowerName.includes("answer") && !answerEntry) {
          answerEntry = zipEntry;
        }
        if (lowerName.endsWith(".txt") && !textEntry) {
          textEntry = zipEntry;
        }
      }
    });
    
    if (!combinedEntry || !answerEntry) {
      throw new Error("Could not find the required image files. Ensure your ZIP contains files with 'combined' and 'answer' in their names.");
    }
    
    return Promise.all([
      combinedEntry.async("blob"),
      answerEntry.async("blob"),
      textEntry ? textEntry.async("string") : Promise.resolve("")
    ]);
  })
  .then(function(results) {
    setupPuzzleViewer(results[0], results[1], results[2]);
  });
}

// Function to set up the puzzle viewer with the extracted files - FIXED
function setupPuzzleViewer(combinedBlob, answerBlob, textContent) {
  const combinedUrl = URL.createObjectURL(combinedBlob);
  const answerUrl = URL.createObjectURL(answerBlob);
  
  // Set the answer image source and store aspect ratio
  const answerImg = document.getElementById('answerImg');
  answerImg.src = answerUrl;
  answerImg.onload = function() {
    answerAspectRatio = answerImg.naturalHeight / answerImg.naturalWidth;
  };
  
  // Show the combined section and toggle button.
  document.getElementById('combinedSection').style.display = "block";
  document.getElementById('toggleAnswerBtn').style.display = "block";
  
  // Ensure tools start at bottom position
  updateToolPosition('bottom');
  
  // ----- Set up Two Canvas Layers for Drawing -----
  const bgCanvas = document.getElementById("bgCanvas");
  const drawCanvas = document.getElementById("drawCanvas");
  const container = document.getElementById("canvasContainer");
  const bgCtx = bgCanvas.getContext("2d");
  const drawCtx = drawCanvas.getContext("2d");
  let imgPainter = new Image();
  imgPainter.onload = function() {
    // Calculate optimal size to utilize more viewing area
    const optimalSize = calculateOptimalCanvasSize(imgPainter.naturalWidth, imgPainter.naturalHeight, true);
    
    // Set container size to be much larger
    container.style.width = optimalSize.width + "px";
    container.style.height = optimalSize.height + "px";
    
    // Set internal resolution to natural dimensions for quality
    bgCanvas.width = imgPainter.naturalWidth;
    bgCanvas.height = imgPainter.naturalHeight;
    drawCanvas.width = imgPainter.naturalWidth;
    drawCanvas.height = imgPainter.naturalHeight;
    
    // Draw the background image
    bgCtx.drawImage(imgPainter, 0, 0);
    
    // Store aspect ratio for resizing
    canvasAspectRatio = imgPainter.naturalHeight / imgPainter.naturalWidth;
    
    // Scroll canvas into view
    container.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  imgPainter.src = combinedUrl;
  
  // ----- Drawing Setup on drawCanvas (Annotation Layer) -----
  let drawing = false;
  let eraserMode = false;
  let currentColor = document.getElementById('colorPicker').value;
  let brushSize = document.getElementById('brushSize').value;
  
  function getMousePos(evt) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY
    };
  }
  
  drawCanvas.addEventListener("mousedown", function(evt) {
    drawing = true;
    const pos = getMousePos(evt);
    drawCtx.beginPath();
    drawCtx.moveTo(pos.x, pos.y);
  });
  
  drawCanvas.addEventListener("mousemove", function(evt) {
    if (!drawing) return;
    const pos = getMousePos(evt);
    if (eraserMode) {
      drawCtx.clearRect(pos.x - brushSize/2, pos.y - brushSize/2, brushSize, brushSize);
    } else {
      drawCtx.lineTo(pos.x, pos.y);
      drawCtx.strokeStyle = currentColor;
      drawCtx.lineWidth = brushSize;
      drawCtx.lineCap = "round";
      drawCtx.stroke();
    }
  });
  
  drawCanvas.addEventListener("mouseup", function() {
    drawing = false;
  });
  
  drawCanvas.addEventListener("mouseleave", function() {
    drawing = false;
  });
  
  // Event listeners for controls
  document.getElementById("colorPicker").addEventListener("change", function() {
    currentColor = this.value;
    eraserMode = false;
    document.getElementById("eraserBtn").textContent = "Eraser";
  });
  
  document.getElementById("brushSize").addEventListener("input", function() {
    brushSize = this.value;
  });
  
  document.getElementById("eraserBtn").addEventListener("click", function() {
    eraserMode = !eraserMode;
    this.textContent = eraserMode ? "Pencil" : "Eraser";
  });
  
  document.getElementById("clearBtn").addEventListener("click", function() {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  });
  
  // Fullscreen toggle
  document.getElementById("fullscreenBtn").addEventListener("click", function() {
    toggleFullscreen();
  });
  
  // Display text file content if present.
  if (textContent) {
    document.getElementById("textPreview").innerText = textContent;
  }
}

// Check for URL parameter on page load
window.addEventListener('DOMContentLoaded', function() {
  const urlParams = new URLSearchParams(window.location.search);
  const viewParam = urlParams.get('view');
  
  if (viewParam) {
    // Show loading indicator
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('zipInput').style.display = 'none';
    
    // Fetch ZIP file from URL
    fetch(viewParam)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.arrayBuffer();
      })
      .then(arrayBuffer => {
        document.getElementById('loadingIndicator').style.display = 'none';
        return processZipFile(arrayBuffer);
      })
      .catch(error => {
        console.error("Error loading ZIP from URL:", error);
        document.getElementById('loadingIndicator').style.display = 'none';
        document.getElementById('zipInput').style.display = 'block';
        document.getElementById('uploadError').textContent = "Error loading ZIP file from URL: " + error.message;
      });
  }
});

// ----- ZIP Extraction and Setup for File Upload -----
document.getElementById('zipInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('uploadError').textContent = "";
  
  processZipFile(file).catch(function(error) {
    console.error("Error reading ZIP file:", error);
    document.getElementById('uploadError').textContent = "Error loading ZIP file: " + error.message;
  });
});

// Settings event listeners - FIXED
document.getElementById('settingsBtn').addEventListener('click', openSettings);
document.getElementById('toolPositionSelect').addEventListener('change', function() {
  updateToolPosition(this.value);
});

// ----- Toggle Answer Key Section -----
function toggleAnswer() {
  const answerSection = document.getElementById('answerSection');
  const toggleBtn = document.getElementById('toggleAnswerBtn');
  if (answerSection.style.display === "none" || answerSection.style.display === "") {
    answerSection.style.display = "block";
    toggleBtn.textContent = "Hide Answer Key";
  } else {
    answerSection.style.display = "none";
    toggleBtn.textContent = "Show Answer Key";
  }
}

// ----- Make the Answer Key Container Draggable and Uniformly Resizable -----
(function() {
  const aContainer = document.getElementById("answerKeyContainer");
  const aHeader = document.getElementById("answerKeyHeader");
  const resizeHandle = document.getElementById("answerResizeHandle");
  let aspectRatio = 1;
  
  // Check if elements exist before setting up event listeners
  if (!aContainer || !aHeader || !resizeHandle) {
    return;
  }
  
  // Dragging functionality
  let posX = 0, posY = 0, startX = 0, startY = 0;
  aHeader.onmousedown = dragMouseDown;
  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }
  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    posX = startX - e.clientX;
    posY = startY - e.clientY;
    startX = e.clientX;
    startY = e.clientY;
    aContainer.style.top = (aContainer.offsetTop - posY) + "px";
    aContainer.style.left = (aContainer.offsetLeft - posX) + "px";
  }
  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
  
  // Resizing functionality with aspect ratio preservation
  resizeHandle.onmousedown = initResize;
  function initResize(e) {
    e.preventDefault();
    aspectRatio = answerAspectRatio || 1; // Use the stored aspect ratio
    document.onmousemove = startResize;
    document.onmouseup = stopResize;
  }
  function startResize(e) {
    const rect = aContainer.getBoundingClientRect();
    let newWidth = e.clientX - rect.left;
    let newHeight = newWidth * aspectRatio;
    
    // Apply minimum size constraints
    if (newWidth < 150) {
      newWidth = 150;
      newHeight = newWidth * aspectRatio;
    }
    
    aContainer.style.width = newWidth + "px";
    aContainer.style.height = newHeight + "px";
  }
  function stopResize() {
    document.onmousemove = null;
    document.onmouseup = null;
  }
})();

// ----- Make the Combined Image Container Uniformly Resizable -----
(function() {
  const container = document.getElementById("canvasContainer");
  const resizeHandle = document.getElementById("canvasResizeHandle");
  
  // Check if elements exist before setting up event listeners
  if (!container || !resizeHandle) {
    return;
  }
  
  resizeHandle.onmousedown = initResize;
  function initResize(e) {
    e.preventDefault();
    document.onmousemove = startResize;
    document.onmouseup = stopResize;
  }
  function startResize(e) {
    const rect = container.getBoundingClientRect();
    let newWidth = e.clientX - rect.left;
    let newHeight = newWidth * canvasAspectRatio;
    
    // Apply minimum size constraints
    if (newWidth < 300) {
      newWidth = 300;
      newHeight = newWidth * canvasAspectRatio;
    }
    
    // Allow expansion beyond normal limits
    const maxWidth = window.innerWidth + 200; // Allow going beyond viewport
    const maxHeight = window.innerHeight + 200;
    
    if (newWidth > maxWidth) {
      newWidth = maxWidth;
      newHeight = newWidth * canvasAspectRatio;
    }
    if (newHeight > maxHeight) {
      newHeight = maxHeight;
      newWidth = newHeight / canvasAspectRatio;
    }
    
    container.style.width = newWidth + "px";
    container.style.height = newHeight + "px";
  }
  function stopResize() {
    document.onmousemove = null;
    document.onmouseup = null;
  }
})();
