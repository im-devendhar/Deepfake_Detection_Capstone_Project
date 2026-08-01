function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return Math.max(1, Math.round(bytes / 1024)) + " KB";
    }
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function showSelectedVideo(file) {
    if (!file) {
        return;
    }

    var $source = $("#video_source");
    var video = $source.parent()[0];
    $source[0].src = URL.createObjectURL(file);
    video.onerror = function () {
        $("#upload-status")
            .text("This video codec cannot be previewed in the browser. It will be analyzed on the server instead.")
            .removeClass("is-error")
            .addClass("is-ready");
    };
    video.onloadeddata = function () {
        $("#upload-status").text("Video ready for analysis.").removeClass("is-error").addClass("is-ready");
    };
    video.load();

    $("#videos").css("display", "block");
    $(".drop-zone").addClass("has-file");
    $("#selected-file-name").text(file.name);
    $("#selected-file-meta").text(formatFileSize(file.size));
    $("#videoUpload").prop("disabled", false);
    $("#upload-status").text("Video ready for analysis.").removeClass("is-error").addClass("is-ready");
}

function getSequenceLength() {
    return Number($("#id_sequence_length").val()) || 20;
}

function drawVideoToCanvas(video, width, height) {
    var canvas = document.createElement("canvas");
    canvas.width = width || video.videoWidth;
    canvas.height = height || video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas;
}

function cropFaceCanvas(frameCanvas, detection) {
    var frameWidth = frameCanvas.width;
    var frameHeight = frameCanvas.height;
    var side;
    var cropX;
    var cropY;

    if (detection) {
        var box = detection.box;
        var centerX = box.x + box.width / 2;
        var centerY = box.y + box.height / 2;
        side = Math.round(Math.max(box.width, box.height) * 1.7);
        cropX = Math.round(centerX - side / 2);
        cropY = Math.round(centerY - side / 2);
    } else {
        side = Math.round(Math.min(frameWidth, frameHeight) * 0.55);
        cropX = Math.round(frameWidth / 2 - side / 2);
        cropY = Math.round(frameHeight * 0.42 - side / 2);
    }

    cropX = Math.max(0, Math.min(cropX, frameWidth - side));
    cropY = Math.max(0, Math.min(cropY, frameHeight - side));

    var cropCanvas = document.createElement("canvas");
    cropCanvas.width = 112;
    cropCanvas.height = 112;
    cropCanvas.getContext("2d").drawImage(frameCanvas, cropX, cropY, side, side, 0, 0, 112, 112);
    return cropCanvas;
}

async function waitForVideoMetadata(video) {
    if (video.readyState >= 1 && video.duration && video.videoWidth) {
        if (video.readyState >= 2) {
            return;
        }
    }

    await new Promise(function (resolve, reject) {
        var done = false;
        var timeout = setTimeout(function () {
            if (!done) {
                done = true;
                reject(new Error("The browser could not decode this video preview in time."));
            }
        }, 5000);
        function finish() {
            if (!done && video.duration && video.videoWidth && video.readyState >= 2) {
                done = true;
                clearTimeout(timeout);
                resolve();
            }
        }
        video.onloadedmetadata = finish;
        video.onloadeddata = finish;
        video.oncanplay = finish;
        video.onerror = function () {
            if (!done) {
                done = true;
                clearTimeout(timeout);
                reject(new Error("The browser cannot decode this video codec."));
            }
        };
        video.load();
    });
}

async function seekVideo(video, time) {
    await new Promise(function (resolve) {
        var timeout = setTimeout(resolve, 1200);
        video.onseeked = function () {
            clearTimeout(timeout);
            resolve();
        };
        video.currentTime = Math.min(Math.max(time, 0.01), Math.max(video.duration - 0.05, 0.01));
    });
}

async function extractClientFaceFrames(file, frameCount, sourceVideo) {
    var faceModelsReady = false;
    try {
        await faceapi.nets.tinyFaceDetector.loadFromUri("/static/json");
        faceModelsReady = true;
    } catch (error) {
        console.warn("Face detector model could not be loaded; using centered crops.", error);
    }

    var video = sourceVideo || document.createElement("video");
    var createdVideo = !sourceVideo;
    if (createdVideo) {
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";
        video.src = URL.createObjectURL(file);
        video.style.left = "-9999px";
        video.style.position = "fixed";
        video.style.top = "0";
        video.style.width = "1px";
        video.style.height = "1px";
        document.body.appendChild(video);
    }

    try {
        await waitForVideoMetadata(video);

        var frames = [];
        for (var index = 0; index < frameCount; index += 1) {
            var seekTime = frameCount === 1 ? 0 : (video.duration * index) / (frameCount - 1);
            await seekVideo(video, seekTime);

            if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
                continue;
            }

            var frameCanvas = drawVideoToCanvas(video);
            var detection = null;
            if (faceModelsReady) {
                try {
                    detection = await faceapi.detectSingleFace(
                        frameCanvas,
                        new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.35 })
                    );
                } catch (error) {
                    console.warn("Face detection failed for one frame; using centered crop.", error);
                }
            }
            var cropCanvas = cropFaceCanvas(frameCanvas, detection);
            frames.push(cropCanvas.toDataURL("image/jpeg", 0.9));
        }

        if (!frames.length) {
            throw new Error("No decoded frames were available for canvas extraction.");
        }
        return frames;
    } finally {
        if (createdVideo) {
            URL.revokeObjectURL(video.src);
            video.remove();
        }
    }
}

$(document).on("change", "#id_upload_video_file", function () {
    showSelectedVideo(this.files[0]);
});

var $dropZone = $("#video-drop-zone");

$dropZone.on("dragenter dragover", function (event) {
    event.preventDefault();
    event.stopPropagation();
    $dropZone.addClass("is-dragging");
});

$dropZone.on("dragleave drop", function (event) {
    event.preventDefault();
    event.stopPropagation();
    $dropZone.removeClass("is-dragging");
});

$dropZone.on("drop", function (event) {
    var files = event.originalEvent.dataTransfer.files;
    if (!files.length) {
        return;
    }

    var input = document.getElementById("id_upload_video_file");
    input.files = files;
    showSelectedVideo(files[0]);
});

$("form").on("submit", async function (event) {
    var form = this;
    if (form.dataset.clientFramesReady === "true") {
        return;
    }

    if (!document.getElementById("id_upload_video_file").files.length) {
        event.preventDefault();
        $("#upload-status").text("Select a video before starting the analysis.").removeClass("is-ready").addClass("is-error");
        return;
    }

    event.preventDefault();
    $("#videoUpload").prop("disabled", true);
    $("#videoUpload").html('Analyzing Video&nbsp;<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span class="sr-only">Loading...</span>');
    $("#upload-status").text("Extracting face frames in the browser. This may take a moment.").removeClass("is-error").addClass("is-ready");

    try {
        var frames = await extractClientFaceFrames(
            document.getElementById("id_upload_video_file").files[0],
            getSequenceLength(),
            document.getElementById("videos")
        );
        $("#client_face_frames").val(JSON.stringify(frames));
        form.dataset.clientFramesReady = "true";
        $("#upload-status").text("Uploading extracted face frames for prediction.").removeClass("is-error").addClass("is-ready");
        HTMLFormElement.prototype.submit.call(form);
    } catch (error) {
        console.error("Browser frame extraction failed", error);
        $("#client_face_frames").val("");
        form.dataset.clientFramesReady = "true";
        $("#upload-status")
            .text("Browser extraction failed; using server extraction for this video.")
            .removeClass("is-error")
            .addClass("is-ready");
        HTMLFormElement.prototype.submit.call(form);
    }
});
