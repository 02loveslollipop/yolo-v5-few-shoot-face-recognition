const CONFIDENCE_THRESHOLD = 0.5;
const CLASSES = ["Juan", "maria", "mateo", "mendoza", "valentina", "roberto", "Jhon", "Santiago", "Miguel", "Juan_A"]; 
const NUM_CLASSES = CLASSES.length;

let session;
const video = document.getElementById('video-cam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const cropCanvas = document.getElementById('crop-canvas');
const cropCtx = cropCanvas.getContext('2d');
const greetingTxt = document.getElementById('greeting-txt');
const statusBadge = document.getElementById('status');

// Create a dedicated off-screen memory canvas just for resizing the tensors for YOLO
// This prevents drawing a black box over the user's live video feed
const tensorCanvas = document.createElement('canvas');
tensorCanvas.width = 640;
tensorCanvas.height = 640;
const tensorCtx = tensorCanvas.getContext('2d', { willReadFrequently: true });

cropCanvas.width = 250;
cropCanvas.height = 250;

async function startApp() {
    // Sanitize the token input by removing any accidental spaces or newlines from copy-pasting
    const tokenInput = document.getElementById('api-token').value.replace(/\s+/g, '');
    const errorMsg = document.getElementById('auth-error');
    const authBtn = document.getElementById('auth-btn');

    if (!tokenInput) {
        errorMsg.innerText = "Please enter an API token.";
        return;
    }

    try {
        errorMsg.innerText = "";
        authBtn.innerText = "Authenticating & Downloading...";
        authBtn.disabled = true;

        statusBadge.innerText = "[1/2] Downloading secure ONNX model...";
        const response = await fetch('./api/model/best.onnx', {
            headers: { 'Authorization': `Bearer ${tokenInput}` }
        });
        
        if (!response.ok) throw new Error("Unauthorized Download");
        const modelBuffer = await response.arrayBuffer();

        document.getElementById('auth-overlay').style.display = 'none';
        
        statusBadge.innerText = "[2/2] Booting ONNX Engine (WebGL)...";
        session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['webgl', 'wasm'] });
        
        statusBadge.innerText = "Active - Secure ONNX Model (WebGL)";
        statusBadge.style.color = "#00ffea";
        
        await setupCamera();
        detectFrame();

    } catch (e) {
        authBtn.innerText = "Unlock Model";
        authBtn.disabled = false;
        console.error(e);
        errorMsg.innerText = "Access Denied: Invalid Token or Server Error.";
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false
    });
    video.srcObject = stream;
    
    // Explicitly force the video to play inline
    video.play();
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            // Dynamically set canvas to match the true camera resolution (e.g. 1920x1080)
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve(video);
        };
    });
}

function processOutputs(res, threshold) {
    const data = res; // Float32Array
    let boxes = [];
    for (let i = 0; i < 25200; i++) {
        const offset = i * 15;
        const boxConf = data[offset + 4]; 
        
        if (boxConf > threshold) {
            let maxClassConf = 0;
            let classId = -1;

            for (let j = 0; j < NUM_CLASSES; j++) {
                const classProb = data[offset + 5 + j];
                if (classProb > maxClassConf) {
                    maxClassConf = classProb;
                    classId = j;
                }
            }

            const overallConf = boxConf * maxClassConf;

            if (overallConf > threshold) {
                const xc = data[offset + 0];
                const yc = data[offset + 1];
                const w = data[offset + 2];
                const h = data[offset + 3];

                const x1 = xc - w / 2;
                const y1 = yc - h / 2;
                boxes.push({ x1, y1, w, h, classId, score: overallConf });
            }
        }
    }
    return boxes.sort((a, b) => b.score - a.score);
}

function preprocessFrame() {
    // Draw the full video frame squashed down to exactly 640x640 on the hidden canvas
    tensorCtx.drawImage(video, 0, 0, 640, 640);
    const imgData = tensorCtx.getImageData(0, 0, 640, 640);
    const data = imgData.data;
    
    // Convert generic RGBA pixel data to properly normalized ONNX Float32 array
    const rgbArray = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        rgbArray[i] = data[i * 4 + 0] / 255.0;            // R
        rgbArray[i + 640 * 640] = data[i * 4 + 1] / 255.0;      // G
        rgbArray[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0;  // B
    }
    return new ort.Tensor('float32', rgbArray, [1, 3, 640, 640]);
}

async function detectFrame() {
    try {
        // Halt processing if the video object hasn't initialized its stream locally yet
        if (video.readyState < 2) {
            requestAnimationFrame(detectFrame);
            return;
        }
        
        const tensor = preprocessFrame();
        
        // Dynamically get input node name rather than guessing 'images'
        const inputName = session.inputNames[0];
        const results = await session.run({ [inputName]: tensor });
        
        const outputName = session.outputNames[0];
        const predictionsArray = results[outputName].data;
        
        const boxes = processOutputs(predictionsArray, CONFIDENCE_THRESHOLD);
        
        // Render 1-to-1 video map onto our visible front-facing canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        let bestDetection = boxes[0] || null;

        boxes.forEach(box => {
            const scaleX = canvas.width / 640;
            const scaleY = canvas.height / 640;
            
            const dx = box.x1 * scaleX;
            const dy = box.y1 * scaleY;
            const dw = box.w * scaleX;
            const dh = box.h * scaleY;
            
            const className = CLASSES[box.classId] || "Unknown";

            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 3;
            ctx.strokeRect(dx, dy, dw, dh);
            
            ctx.fillStyle = "#00ff00";
            ctx.fillRect(dx, dy - 25, ctx.measureText(`${className} ${Math.round(box.score*100)}%`).width + 10, 25);
            ctx.fillStyle = "#000000";
            ctx.font = "18px Arial bold";
            ctx.fillText(`${className} ${Math.round(box.score*100)}%`, dx + 5, dy - 5);
        });

        if (bestDetection) {
            const className = CLASSES[bestDetection.classId] || "Unknown";
            greetingTxt.innerText = `Hello ${className}!`;
            greetingTxt.style.color = "#00ffea";
            
            const scaleX = canvas.width / 640;
            const scaleY = canvas.height / 640;
            
            let cx = Math.max(0, bestDetection.x1 * scaleX);
            let cy = Math.max(0, bestDetection.y1 * scaleY);
            let cw = Math.min(canvas.width - cx, bestDetection.w * scaleX);
            let ch = Math.min(canvas.height - cy, bestDetection.h * scaleY);
            
            try {
                cropCtx.drawImage(canvas, cx, cy, cw, ch, 0, 0, cropCanvas.width, cropCanvas.height);
            } catch (e) {}
        } else {
            greetingTxt.innerText = "...";
            greetingTxt.style.color = "#666";
        }

    } catch (e) {
        console.error("Frame processing error:", e);
        statusBadge.innerText = "Error: " + e.message;
        statusBadge.style.color = "red";
    }

    // Schedule next frame
    requestAnimationFrame(detectFrame);
}

document.getElementById('auth-btn').onclick = startApp;
