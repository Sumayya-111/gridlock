
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { detectImage, streamVideoDetection } from '../api/client';
import './DetectionPanel.css';

export default function DetectionPanel({ settings, onViolationsUpdate }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [currentViolations, setCurrentViolations] = useState([]);
  const [progress, setProgress] = useState(0);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [liveFrameId, setLiveFrameId] = useState(0);
  const [violationCount, setViolationCount] = useState(0);

  // Video preview state
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewReady, setPreviewReady] = useState(false);
  const [frameSize, setFrameSize] = useState(null);

  // Calibration state
  const [calibrationMode, setCalibrationMode] = useState(null);
  const [signalPoints, setSignalPoints] = useState([]);
  const [linePoints, setLinePoints] = useState([]);
  const [wrongWayPoints, setWrongWayPoints] = useState([]);
  const [wrongWayDirection, setWrongWayDirection] = useState('down');

  const streamRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Create preview URL for selected video.
  useEffect(() => {
    if (!file || settings.inputMode !== 'video') {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file, settings.inputMode]);

  // Close WebSocket when component is removed.
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  // Draw video preview and all calibration markings.
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !previewReady || !frameSize) return;

    canvas.width = frameSize.width;
    canvas.height = frameSize.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw traffic-light area.
    if (signalPoints.length > 0) {
      const [first, second] = signalPoints;

      ctx.fillStyle = 'rgba(0, 200, 100, 0.15)';
      ctx.strokeStyle = '#00e676';
      ctx.lineWidth = Math.max(2, canvas.width / 500);

      if (second) {
        const x = Math.min(first.x, second.x);
        const y = Math.min(first.y, second.y);
        const width = Math.abs(second.x - first.x);
        const height = Math.abs(second.y - first.y);

        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);
      } else {
        ctx.beginPath();
        ctx.arc(first.x, first.y, 7, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw stop line.
    if (linePoints.length > 0) {
      ctx.strokeStyle = '#ff5252';
      ctx.lineWidth = Math.max(3, canvas.width / 400);
      ctx.beginPath();
      ctx.moveTo(linePoints[0].x, linePoints[0].y);

      if (linePoints[1]) {
        ctx.lineTo(linePoints[1].x, linePoints[1].y);
      }

      ctx.stroke();

      linePoints.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ff5252';
        ctx.fill();
      });
    }

    // Draw wrong-way road zone.
    if (wrongWayPoints.length > 0) {
      const first = wrongWayPoints[0];
      const second = wrongWayPoints[1];

      ctx.strokeStyle = '#ffb300';
      ctx.fillStyle = 'rgba(255, 179, 0, 0.15)';
      ctx.lineWidth = Math.max(3, canvas.width / 400);

      if (second) {
        const x = Math.min(first.x, second.x);
        const y = Math.min(first.y, second.y);
        const width = Math.abs(second.x - first.x);
        const height = Math.abs(second.y - first.y);

        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);
      } else {
        ctx.beginPath();
        ctx.arc(first.x, first.y, 7, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [
    previewReady,
    frameSize,
    signalPoints,
    linePoints,
    wrongWayPoints,
  ]);

  // Read video dimensions and match backend frame resizing.
  const handleVideoLoaded = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth || !video.videoHeight) return;

    const scale = video.videoWidth > 1280
      ? 1280 / video.videoWidth
      : 1;

    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);

    setFrameSize({ width, height });
    setPreviewReady(true);
  };

  // Handle calibration clicks on the video canvas.
  const handleCanvasClick = (event) => {
    if (!calibrationMode || !canvasRef.current || !frameSize) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const point = {
      x: Math.max(
        0,
        Math.min(
          canvas.width,
          Math.round(
            ((event.clientX - rect.left) / rect.width) * canvas.width
          )
        )
      ),
      y: Math.max(
        0,
        Math.min(
          canvas.height,
          Math.round(
            ((event.clientY - rect.top) / rect.height) * canvas.height
          )
        )
      ),
    };

    if (calibrationMode === 'signal') {
      setSignalPoints((previous) => [...previous, point].slice(-2));

      if (signalPoints.length >= 1) {
        setCalibrationMode(null);
      }
    }

    if (calibrationMode === 'line') {
      setLinePoints((previous) => [...previous, point].slice(-2));

      if (linePoints.length >= 1) {
        setCalibrationMode(null);
      }
    }

    if (calibrationMode === 'wrongway') {
      setWrongWayPoints((previous) => [...previous, point].slice(-2));

      if (wrongWayPoints.length >= 1) {
        setCalibrationMode(null);
      }
    }
  };

  const stopProcessing = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    setProcessingVideo(false);
    setStatusMessage('Stopped by user.');
  }, []);

  // Process a single image.
  const processImage = async (imageFile) => {
    setLoading(true);
    setError(null);
    setStatusMessage('Uploading image for detection...');

    try {
      const data = await detectImage(imageFile, settings);

      setResultImage(data.annotated_image_b64);
      setCurrentViolations(data.violations || []);
      onViolationsUpdate?.(data.violations || []);
      setStatusMessage('Image processed successfully.');
    } catch (err) {
      setError(err.message);
      setStatusMessage('Image processing failed.');
    } finally {
      setLoading(false);
    }
  };

  // Convert calibration points and start video detection.
  const startVideoProcessing = (videoFile = file) => {
    if (!videoFile) return;

    let signalROI = null;
    let stopLine = null;
    let wrongWay = null;

    // Traffic-light rectangle.
    if (signalPoints.length === 2) {
      const x1 = Math.min(signalPoints[0].x, signalPoints[1].x);
      const y1 = Math.min(signalPoints[0].y, signalPoints[1].y);
      const x2 = Math.max(signalPoints[0].x, signalPoints[1].x);
      const y2 = Math.max(signalPoints[0].y, signalPoints[1].y);

      signalROI = [x1, y1, x2, y2];
    }

    // Stop-line endpoints.
    if (linePoints.length === 2) {
      stopLine = [
        linePoints[0].x,
        linePoints[0].y,
        linePoints[1].x,
        linePoints[1].y,
      ];
    }

    // Wrong-way road rectangle and allowed direction.
    if (wrongWayPoints.length === 2) {
      const x1 = Math.min(wrongWayPoints[0].x, wrongWayPoints[1].x);
      const y1 = Math.min(wrongWayPoints[0].y, wrongWayPoints[1].y);
      const x2 = Math.max(wrongWayPoints[0].x, wrongWayPoints[1].x);
      const y2 = Math.max(wrongWayPoints[0].y, wrongWayPoints[1].y);

      const directionVectors = {
        up: [0, -1],
        down: [0, 1],
        left: [-1, 0],
        right: [1, 0],
      };

      const [dx, dy] = directionVectors[wrongWayDirection];

      wrongWay = [x1, y1, x2, y2, dx, dy];
    }

    const calibration = {
      signalROI,
      stopLine,
      wrongWay,
    };

    setProcessingVideo(true);
    setError(null);
    setResultImage(null);
    setStatusMessage('Uploading video — detection starting...');
    setProgress(0);
    setCurrentViolations([]);
    setViolationCount(0);

    let totalViolations = 0;

    streamRef.current = streamVideoDetection(
      videoFile,
      settings,
      (data) => {
        if (data.annotated_image_b64) {
          setResultImage(data.annotated_image_b64);
        }

        if (data.frame_id !== undefined) {
          setLiveFrameId(data.frame_id);
        }

        if (data.violations && data.violations.length > 0) {
          totalViolations += data.violations.length;
          setViolationCount(totalViolations);

          setCurrentViolations((previous) =>
            [...previous, ...data.violations].slice(-20)
          );

          onViolationsUpdate?.(data.violations);
        }

        if (data.progress !== undefined) {
          const percentage = Math.round(data.progress * 100);
          setProgress(percentage);

          setStatusMessage(
            `Scanning... ${percentage}% — ${totalViolations} violation(s) found`
          );
        }
      },
      () => {
        setProcessingVideo(false);
        setProgress(100);

        setStatusMessage(
          `Done — ${totalViolations} violation(s) detected.`
        );

        streamRef.current = null;
      },
      (err) => {
        setError(err.message || 'Video processing failed');
        setProcessingVideo(false);
        setStatusMessage('Video processing failed.');
        streamRef.current = null;
      },
      calibration
    );
  };

  // Upload handler.
  const handleFileChange = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    setFile(selectedFile);
    setError(null);
    setResultImage(null);
    setCurrentViolations([]);
    setProgress(0);
    setLiveFrameId(0);
    setViolationCount(0);
    setStatusMessage('');

    setSignalPoints([]);
    setLinePoints([]);
    setWrongWayPoints([]);
    setWrongWayDirection('down');
    setCalibrationMode(null);

    setPreviewReady(false);
    setFrameSize(null);

    if (settings.inputMode === 'image') {
      await processImage(selectedFile);
    } else if (settings.inputMode === 'video') {
      setStatusMessage(
        'Video loaded. Configure the detection zones before starting.'
      );
    }
  };

  // Reset the entire interface.
  const resetAll = () => {
    stopProcessing();

    setFile(null);
    setResultImage(null);
    setCurrentViolations([]);
    setProgress(0);
    setLiveFrameId(0);
    setViolationCount(0);
    setStatusMessage('');
    setError(null);

    setSignalPoints([]);
    setLinePoints([]);
    setWrongWayPoints([]);
    setWrongWayDirection('down');
    setCalibrationMode(null);

    setPreviewReady(false);
    setFrameSize(null);
  };

  const redLightCalibrationComplete =
    signalPoints.length === 2 && linePoints.length === 2;

  const wrongWayCalibrationComplete =
    wrongWayPoints.length === 2;

  return (
    <div className="detection-panel">
      <div className="panel-header">
        <div className="panel-title govt-badge">
          Detection Interface
        </div>

        {processingVideo && (
          <button className="btn-stop" onClick={stopProcessing}>
            STOP
          </button>
        )}

        {(resultImage || error) && !processingVideo && (
          <button className="reset-btn" onClick={resetAll}>
            UPLOAD ANOTHER
          </button>
        )}
      </div>

      {/* Upload */}
      {!file && !loading && (
        <div className="upload-area">
          <label className="file-input-label">
            <div className="upload-icon">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>

            <div className="upload-text">
              Upload {settings.inputMode === 'image' ? 'Image' : 'Video'}
            </div>

            <div className="upload-subtext">
              Click to select a file
            </div>

            <input
              type="file"
              accept={
                settings.inputMode === 'image'
                  ? 'image/*'
                  : 'video/mp4,video/x-m4v,video/*'
              }
              onChange={handleFileChange}
            />
          </label>
        </div>
      )}

      {/* Hidden video for calibration preview */}
      {previewUrl && (
        <video
          ref={videoRef}
          src={previewUrl}
          preload="auto"
          muted
          playsInline
          onLoadedData={handleVideoLoaded}
          style={{ display: 'none' }}
        />
      )}

      {/* Video calibration interface */}
      {file &&
        settings.inputMode === 'video' &&
        !processingVideo &&
        !resultImage && (
          <div
            className="calibration-panel"
            style={{
              margin: '20px 0',
              padding: '16px',
              border: '1px solid #444',
              borderRadius: '10px',
            }}
          >
            <h3>Camera Calibration</h3>

            <p>
              Select detection areas on the first frame of your video.
              The coordinates will match the backend video frame.
            </p>

            {!previewReady && (
              <p>Loading video preview...</p>
            )}

            {previewReady && (
              <>
                <div style={{ marginBottom: '12px' }}>
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    style={{
                      display: 'block',
                      width: '100%',
                      height: 'auto',
                      maxWidth: '100%',
                      borderRadius: '6px',
                      cursor: calibrationMode ? 'crosshair' : 'default',
                    }}
                  />
                </div>

                <p style={{ fontSize: '13px' }}>
                  Frame size: {frameSize.width} × {frameSize.height}
                </p>

                {/* Red-light calibration */}
                <div
                  style={{
                    borderTop: '1px solid #555',
                    paddingTop: '12px',
                    marginTop: '12px',
                  }}
                >
                  <h4>Red-Light Calibration</h4>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                    <button
                      type="button"
                      onClick={() => {
                        setSignalPoints([]);
                        setCalibrationMode('signal');
                      }}
                    >
                      Select Traffic Light
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setLinePoints([]);
                        setCalibrationMode('line');
                      }}
                    >
                      Select Stop Line
                    </button>
                  </div>

                  {calibrationMode === 'signal' && (
                    <p style={{ color: '#00e676' }}>
                      Click two opposite corners around the traffic light.
                    </p>
                  )}

                  {calibrationMode === 'line' && (
                    <p style={{ color: '#ff5252' }}>
                      Click the two endpoints of the stop line.
                    </p>
                  )}

                  <p style={{ fontSize: '13px' }}>
                    Traffic light: {signalPoints.length === 2 ? 'Selected ✓' : 'Not selected'}
                  </p>

                  <p style={{ fontSize: '13px' }}>
                    Stop line: {linePoints.length === 2 ? 'Selected ✓' : 'Not selected'}
                  </p>
                </div>

                {/* Wrong-way calibration */}
                <div
                  style={{
                    borderTop: '1px solid #555',
                    paddingTop: '12px',
                    marginTop: '16px',
                  }}
                >
                  <h4>Wrong-Way Calibration</h4>

                  <p>
                    Select the road area where vehicles should travel,
                    then choose the normal direction of traffic.
                  </p>

                  <button
                    type="button"
                    onClick={() => {
                      setWrongWayPoints([]);
                      setCalibrationMode('wrongway');
                    }}
                  >
                    Select Road Zone
                  </button>

                  {calibrationMode === 'wrongway' && (
                    <p style={{ color: '#ffb300' }}>
                      Click two opposite corners to mark the road area.
                    </p>
                  )}

                  <div style={{ marginTop: '12px' }}>
                    <label
                      htmlFor="wrong-way-direction"
                      style={{ display: 'block', marginBottom: '6px' }}
                    >
                      Allowed traffic direction:
                    </label>

                    <select
                      id="wrong-way-direction"
                      value={wrongWayDirection}
                      onChange={(event) =>
                        setWrongWayDirection(event.target.value)
                      }
                      style={{
                        padding: '8px',
                        minWidth: '180px',
                      }}
                    >
                      <option value="up">Up (towards top)</option>
                      <option value="down">Down (towards bottom)</option>
                      <option value="left">Left (towards left)</option>
                      <option value="right">Right (towards right)</option>
                    </select>
                  </div>

                  <p style={{ fontSize: '13px' }}>
                    Road zone: {wrongWayCalibrationComplete ? 'Selected ✓' : 'Not selected'}
                  </p>

                  <p style={{ fontSize: '12px', opacity: 0.8 }}>
                    Choose the direction in which normal traffic travels
                    in this camera view, not the direction of a wrong-way vehicle.
                  </p>
                </div>

                {/* Clear calibration */}
                <button
                  type="button"
                  onClick={() => {
                    setSignalPoints([]);
                    setLinePoints([]);
                    setWrongWayPoints([]);
                    setWrongWayDirection('down');
                    setCalibrationMode(null);
                  }}
                  style={{ marginTop: '16px' }}
                >
                  Clear All Calibration
                </button>

                <p style={{ fontSize: '12px', opacity: 0.8 }}>
                  Calibration depends on the camera angle. If the camera
                  position changes, calibrate again.
                </p>
              </>
            )}

            {/* Start and cancel */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '10px',
                marginTop: '16px',
              }}
            >
              <button
                type="button"
                onClick={() => startVideoProcessing(file)}
                disabled={!previewReady || processingVideo || calibrationMode !== null}
              >
                Start Detection
              </button>

              <button type="button" onClick={resetAll}>
                Cancel
              </button>
            </div>

            {previewReady && (
              <div style={{ fontSize: '12px', marginTop: '10px' }}>
                {!redLightCalibrationComplete && (
                  <p>
                    Red-light detection is disabled until both the traffic
                    light and stop line are selected.
                  </p>
                )}

                {!wrongWayCalibrationComplete && (
                  <p>
                    Wrong-way detection is disabled until a road zone is selected.
                  </p>
                )}

                <p>
                  Other configured detection modules can run without these
                  calibrations.
                </p>
              </div>
            )}
          </div>
        )}

      {/* Image loading */}
      {loading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <div className="loading-text">Analyzing frame...</div>
        </div>
      )}

      {/* Live scanning header */}
      {processingVideo && (
        <div className="live-scan-header">
          <div className="live-badge">
            <span className="live-dot"></span>
            LIVE SCAN
          </div>

          <div className="live-stats">
            <span className="live-stat">
              Frame <strong>{liveFrameId}</strong>
            </span>

            <span className="live-stat violations-stat">
              <strong>{violationCount}</strong> violation(s)
            </span>
          </div>
        </div>
      )}

      {/* Annotated output */}
      {resultImage && (
        <div className="result-container">
          <img
            src={`data:image/jpeg;base64,${resultImage}`}
            alt="Annotated Result"
            className="result-image"
          />
        </div>
      )}

      {/* Video progress */}
      {processingVideo && (
        <div className="progress-state">
          <div className="progress-info">
            <span className="progress-label">Processing</span>
            <span className="progress-pct">{Math.round(progress)}%</span>
          </div>

          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {statusMessage && (
        <div className="status-box">
          <div className="status-text">{statusMessage}</div>
        </div>
      )}

      {error && (
        <div className="error-box">
          <div className="error-icon">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>

          <div className="error-text">{error}</div>
        </div>
      )}
    </div>
  );
}
