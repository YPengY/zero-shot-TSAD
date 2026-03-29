/**
 * Workbench-local DOM bindings and mutable UI state.
 *
 * The runtime module owns only element lookup and state-shape definition. It
 * does not fetch data, render charts, or implement job orchestration.
 */

/**
 * Collect all DOM handles used by the workbench controller and renderers.
 */
export function createWorkbenchDom() {
  return {
    previewBatch: document.getElementById("preview-batch"),
    previewCount: document.getElementById("wb-preview-count"),
    previewSeed: document.getElementById("wb-preview-seed"),
    previewGenerate: document.getElementById("wb-preview-generate"),
    previewGallery: document.getElementById("wb-preview-gallery"),
    runName: document.getElementById("wb-run-name"),
    trainSamples: document.getElementById("wb-train-samples"),
    valSamples: document.getElementById("wb-val-samples"),
    testSamples: document.getElementById("wb-test-samples"),
    samplesPerShard: document.getElementById("wb-samples-per-shard"),
    seedBase: document.getElementById("wb-seed-base"),
    trainTemplate: document.getElementById("wb-train-template"),
    trainDevice: document.getElementById("wb-train-device"),
    trainEpochs: document.getElementById("wb-train-epochs"),
    trainBatchSize: document.getElementById("wb-train-batch-size"),
    evalBatchSize: document.getElementById("wb-eval-batch-size"),
    overwriteRun: document.getElementById("wb-overwrite-run"),
    generateSubmit: document.getElementById("wb-generate-submit"),
    loadRun: document.getElementById("wb-load-run"),
    runSummary: document.getElementById("wb-run-summary"),
    runPath: document.getElementById("wb-run-path"),
    splitSelect: document.getElementById("wb-split-select"),
    sampleSelect: document.getElementById("wb-sample-select"),
    featureIndex: document.getElementById("wb-feature-index"),
    sliceStart: document.getElementById("wb-slice-start"),
    sliceEnd: document.getElementById("wb-slice-end"),
    contextSize: document.getElementById("wb-context-size"),
    stride: document.getElementById("wb-stride"),
    patchSize: document.getElementById("wb-patch-size"),
    windowIndex: document.getElementById("wb-window-index"),
    loadSamples: document.getElementById("wb-load-samples"),
    previewPackedSample: document.getElementById("wb-preview-packed-sample"),
    previewWindow: document.getElementById("wb-preview-window"),
    sampleStats: document.getElementById("wb-sample-stats"),
    sampleCanvas: document.getElementById("wb-sample-canvas"),
    windowCanvas: document.getElementById("wb-window-canvas"),
    patchCanvas: document.getElementById("wb-patch-canvas"),
    datasetJson: document.getElementById("wb-dataset-json"),
    trainConfigPath: document.getElementById("wb-train-config-path"),
    trainOutputDir: document.getElementById("wb-train-output-dir"),
    trainDeviceOverride: document.getElementById("wb-train-device-override"),
    trainEpochOverride: document.getElementById("wb-train-epoch-override"),
    inspectData: document.getElementById("wb-inspect-data"),
    inspectMaxSamples: document.getElementById("wb-inspect-max-samples"),
    startTraining: document.getElementById("wb-start-training"),
    refreshTraining: document.getElementById("wb-refresh-training"),
    trainingStatus: document.getElementById("wb-training-status"),
    trainingStage: document.getElementById("wb-training-stage"),
    trainingProgressFill: document.getElementById("wb-training-progress-fill"),
    trainingProgressMeta: document.getElementById("wb-training-progress-meta"),
    trainingKpis: document.getElementById("wb-training-kpis"),
    lossCanvas: document.getElementById("wb-loss-canvas"),
    qualityCanvas: document.getElementById("wb-quality-canvas"),
    calibrationCanvas: document.getElementById("wb-calibration-canvas"),
    lossLegend: document.getElementById("wb-loss-legend"),
    qualityLegend: document.getElementById("wb-quality-legend"),
    calibrationLegend: document.getElementById("wb-calibration-legend"),
    trainingSummary: document.getElementById("wb-training-summary"),
    trainingLog: document.getElementById("wb-training-log"),
    datasetWarning: document.getElementById("wb-dataset-warning"),
  };
}

/**
 * Create the mutable state container shared by workbench submodules.
 */
export function createWorkbenchState() {
  return {
    bootstrap: {},
    previewCards: [],
    activePreviewId: null,
    runInfo: null,
    sampleList: [],
    currentSample: null,
    currentWindow: null,
    activeInspector: null,
    trainingJob: null,
    trainingMetrics: null,
    polling: { generate: null, train: null },
  };
}
