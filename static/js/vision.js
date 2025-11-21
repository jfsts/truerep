import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

// UI Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const repCountElement = document.getElementById("rep-count");
const statusElement = document.getElementById("status-indicator");
const angleDebugElement = document.getElementById("angle-debug");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");

let poseLandmarker = undefined;
let runningMode = "VIDEO";
let lastVideoTime = -1;
let results = undefined;

// Application State
let isCounting = false;
let squatState = "UP"; // "UP" or "DOWN"
let squatCount = 0;
let squatHoldFrames = 0; // Frames held in squat position
let standHoldFrames = 0; // Frames held in standing position
let hasConfirmedStand = false; // Require user to be standing before starting cycle
let hasCompletedFirstSquat = false; // Ensure we don't count until first DOWN is reached

// Constants
const ANGLE_STANDING_THRESHOLD = 165; // Degrees for standing
const ANGLE_SQUAT_THRESHOLD = 90; // Degrees for squatting (agachamento completo - coxas paralelas)
const VISIBILITY_THRESHOLD = 0.60; // Surgical adjustment: 0.60
const MIN_BODY_HEIGHT = 0.4; // Surgical adjustment: 0.4 (40% of screen height)
const REQUIRED_HOLD_FRAMES = 15; // Number of frames to confirm state change (~0.5s at 30fps)

// Initialize MediaPipe Pose Landmarker
const createPoseLandmarker = async () => {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                delegate: "GPU"
            },
            runningMode: runningMode,
            numPoses: 1,
            minPoseDetectionConfidence: 0.6,
            minPosePresenceConfidence: 0.6,
            minTrackingConfidence: 0.6
        });
        console.log("PoseLandmarker initialized successfully");
    } catch (error) {
        console.error("Error initializing PoseLandmarker:", error);
        statusElement.innerText = "Erro ao carregar IA";
        statusElement.classList.add("text-red-500");
    }
};

// Start Webcam
const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Seu navegador não suporta acesso à câmera. Use Chrome ou Safari.");
        return;
    }

    // Detectar se é mobile
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    try {
        // Primeira tentativa: câmera frontal em mobile
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: isMobile ? "user" : "environment"
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        console.log("Câmera iniciada com sucesso");
    } catch (error) {
        console.error("Erro ao acessar câmera:", error);

        // Fallback: tentar qualquer câmera disponível
        try {
            const fallbackStream = await navigator.mediaDevices.getUserMedia({
                video: true
            });
            video.srcObject = fallbackStream;
            video.addEventListener("loadeddata", predictWebcam);
            console.log("Câmera iniciada com fallback");
        } catch (fallbackError) {
            console.error("Erro no fallback:", fallbackError);
            alert("Não foi possível acessar a câmera.\n\n" +
                "SOLUÇÃO: Use Chrome no celular e permita acesso à câmera.\n\n" +
                "Nota: Alguns navegadores bloqueiam câmera em HTTP.\n" +
                "Se não funcionar, teste no computador primeiro.");
        }
    }
};

// Calculate Angle between 3 points (A, B, C) -> Angle at B
function calculateAngle(a, b, c) {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
}

// Check if full body is visible and large enough
function isFullBodyVisible(landmarks) {
    // Key landmarks: Shoulders (11, 12), Hips (23, 24), Knees (25, 26), Ankles (27, 28)
    const keyLandmarksIndices = [11, 12, 23, 24, 25, 26, 27, 28];
    let minY = 1, maxY = 0;

    for (const index of keyLandmarksIndices) {
        const lm = landmarks[index];
        // Check visibility score
        if (lm.visibility < VISIBILITY_THRESHOLD) {
            return { visible: false, reason: "low_visibility" };
        }
        // Check bounds (must be within frame)
        if (lm.x < 0 || lm.x > 1 || lm.y < 0 || lm.y > 1) {
            return { visible: false, reason: "out_of_bounds" };
        }

        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
    }

    // Check Body Size (Height)
    const bodyHeight = maxY - minY;
    if (bodyHeight < MIN_BODY_HEIGHT) {
        return { visible: false, reason: "too_small" };
    }

    return { visible: true };
}

// Main Prediction Loop
async function predictWebcam() {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (poseLandmarker && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        results = poseLandmarker.detectForVideo(video, performance.now());
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results && results.landmarks) {
        for (const landmarks of results.landmarks) {
            // Draw landmarks
            const drawingUtils = new DrawingUtils(canvasCtx);

            // Draw Connectors (Skeleton) - Green and Thick
            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 4
            });

            // Draw Landmarks (Points) - Red with White Border
            drawingUtils.drawLandmarks(landmarks, {
                radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
                color: '#FF0000',
                lineWidth: 2,
                fillColor: '#FFFFFF'
            });

            // Draw Bounding Box
            const boundingBox = getBoundingBox(landmarks);
            drawBoundingBox(boundingBox);

            // Process Logic
            processSquatLogic(landmarks);
        }
    }
    canvasCtx.restore();

    window.requestAnimationFrame(predictWebcam);
}

function processSquatLogic(landmarks) {
    // 1. Check Full Body Visibility
    const visibilityCheck = isFullBodyVisible(landmarks);
    if (!visibilityCheck.visible) {
        if (visibilityCheck.reason === "too_small") {
            statusElement.innerText = "APROXIME-SE";
            statusElement.className = "text-xl font-bold text-yellow-500";
        } else {
            statusElement.innerText = "CORPO INTEIRO NÃO VISÍVEL";
            statusElement.className = "text-xl font-bold text-red-500";
        }
        angleDebugElement.innerText = "-";
        return; // Stop processing
    }

    // 2. Calculate Hip Extension Angle
    const angleLeft = calculateAngle(landmarks[11], landmarks[23], landmarks[25]);
    const angleRight = calculateAngle(landmarks[12], landmarks[24], landmarks[26]);
    const avgHipAngle = (angleLeft + angleRight) / 2;

    // Update Debug UI (Show Angle and Internal State)
    angleDebugElement.innerText = `${Math.round(avgHipAngle)}° | ${squatState}`;

    // 3. Squat State Machine & Counting
    if (isCounting) {
        // INITIAL CHECK: Require user to be standing before starting the cycle
        if (!hasConfirmedStand) {
            if (avgHipAngle > ANGLE_STANDING_THRESHOLD) {
                standHoldFrames++;
                if (standHoldFrames >= REQUIRED_HOLD_FRAMES) {
                    hasConfirmedStand = true;
                    standHoldFrames = 0; // Reset for normal logic
                    squatState = "UP";
                }
            } else {
                standHoldFrames = 0;
            }

            statusElement.innerText = "FIQUE EM PÉ";
            statusElement.className = "text-xl font-bold text-blue-500";
            return; // Wait until standing is confirmed
        }

        // Debouncing Logic
        if (avgHipAngle < ANGLE_SQUAT_THRESHOLD) {
            squatHoldFrames++;
            standHoldFrames = 0;
        } else if (avgHipAngle > ANGLE_STANDING_THRESHOLD) {
            standHoldFrames++;
            squatHoldFrames = 0;
        } else {
            // In transition zone, reset counters
            squatHoldFrames = 0;
            standHoldFrames = 0;
        }

        // State Transitions
        if (squatState === "UP") {
            if (squatHoldFrames >= REQUIRED_HOLD_FRAMES) {
                squatState = "DOWN";
                hasCompletedFirstSquat = true; // Mark that we've seen at least one squat
            }
        } else if (squatState === "DOWN") {
            if (standHoldFrames >= REQUIRED_HOLD_FRAMES) {
                squatState = "UP";
                // Only count if we've completed at least one squat (gone DOWN first)
                if (hasCompletedFirstSquat) {
                    squatCount++;
                    repCountElement.innerText = squatCount;
                }
            }
        }

        // Update Status Display (Only when counting)
        let displayState = "EM PÉ";
        let colorClass = "text-green-500";

        // Visual feedback follows the STATE now, not just the angle, for consistency
        if (squatState === "DOWN") {
            displayState = "AGACHADO";
            colorClass = "text-orange-500";
        } else {
            displayState = "EM PÉ";
            colorClass = "text-green-500";
        }

        // Override visual if angle is clearly squatting but state hasn't swapped yet (optional, but keeps UI snappy)
        if (avgHipAngle < ANGLE_SQUAT_THRESHOLD) {
            displayState = "AGACHADO";
            colorClass = "text-orange-500";
        }

        statusElement.innerText = displayState;
        statusElement.className = `text-xl font-bold ${colorClass}`;

    } else {
        // Not counting -> Show PAUSADO
        statusElement.innerText = "PAUSADO";
        statusElement.className = "text-xl font-bold text-yellow-400";
    }
}

function getBoundingBox(landmarks) {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const lm of landmarks) {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
    }
    return { minX, minY, maxX, maxY };
}

function drawBoundingBox(bbox) {
    const width = canvasElement.width;
    const height = canvasElement.height;

    canvasCtx.strokeStyle = "#00FF00";
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(
        bbox.minX * width,
        bbox.minY * height,
        (bbox.maxX - bbox.minX) * width,
        (bbox.maxY - bbox.minY) * height
    );
}

// Control Functions
function startCounting() {
    isCounting = true;
    hasConfirmedStand = false; // Reset this to force a re-check
    hasCompletedFirstSquat = false; // Reset this too - require a full squat before counting
    standHoldFrames = 0;
    squatHoldFrames = 0;
    btnStart.classList.add("hidden");
    btnStop.classList.remove("hidden");
    // Status will be updated by the loop immediately
}

function stopCounting() {
    isCounting = false;
    btnStart.classList.remove("hidden");
    btnStop.classList.add("hidden");
    statusElement.innerText = "PAUSADO";
    statusElement.className = "text-xl font-bold text-yellow-400";
}

function resetCounting() {
    squatCount = 0;
    repCountElement.innerText = squatCount;
    squatState = "UP";
    hasConfirmedStand = false; // Reset this too
    hasCompletedFirstSquat = false; // Reset this too
    standHoldFrames = 0;
    squatHoldFrames = 0;
    stopCounting();
}

// Event Listeners
btnStart.addEventListener("click", startCounting);
btnStop.addEventListener("click", stopCounting);
btnReset.addEventListener("click", resetCounting);

// Start the app
startCamera();
createPoseLandmarker();
