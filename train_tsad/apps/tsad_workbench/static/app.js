/**
 * Main browser shell for TSAD Studio and the embedded workbench.
 *
 * This file owns page-level composition, localization tables, and shared
 * helpers that are consumed by the preview surface and the workbench modules.
 * Domain-specific rendering and controller logic live in dedicated submodules.
 */

const LOCALE_TEXT = {
  en: {
    "brand.eyebrow": "TSAD Studio",
    "brand.title": "Interactive Config and Sample Preview",
    "brand.subtitle": "Isolated workspace for parameter editing, randomization, and in-memory visualization.",
    "toolbar.import": "Import Config",
    "toolbar.restore": "Restore Defaults",
    "toolbar.randomize": "Random Fill All",
    "toolbar.preview": "Preview Sample",
    "toolbar.previewBatch": "Preview Gallery",
    "workbench.previewRail": "Preview Workflow",
    "workbench.previewGalleryTitle": "Preview Gallery",
    "workbench.previewGallerySubtitle": "Generate multiple in-memory samples from the current config, then open any one in the main preview panel.",
    "workbench.previewCount": "Preview Count",
    "workbench.previewSeed": "Base Seed",
    "workbench.previewGenerate": "Build Preview Gallery",
    "workbench.generationRail": "Dataset Workflow",
    "workbench.generateTitle": "Generate And Pack Dataset",
    "workbench.generateSubtitle": "The current edited config will be written as the runtime synthetic config, the train/val/test splits will be generated in parallel, then packed and linked to a train config.",
    "workbench.runName": "Run Name",
    "workbench.trainSamples": "Train Samples",
    "workbench.valSamples": "Val Samples",
    "workbench.testSamples": "Test Samples",
    "workbench.shardSize": "Samples Per Shard",
    "workbench.seedBase": "Seed Base",
    "workbench.trainTemplate": "Train Template",
    "workbench.trainDevice": "Train Device",
    "workbench.trainEpochs": "Train Epochs",
    "workbench.trainBatchSize": "Train Batch Size",
    "workbench.evalBatchSize": "Eval Batch Size",
    "workbench.overwriteRun": "Overwrite existing run directory if present",
    "workbench.generateAction": "Generate + Pack",
    "workbench.loadRun": "Load Existing Run",
    "workbench.datasetRail": "Dataset Inspector",
    "workbench.datasetTitle": "Packed Sample And Window Viewer",
    "workbench.datasetSubtitle": "Browse packed samples, inspect sequence slices, and visualize the exact training windows and patch labels.",
    "workbench.runPath": "Run Or Packed Root",
    "workbench.split": "Split",
    "workbench.sample": "Sample",
    "workbench.feature": "Feature",
    "workbench.sliceStart": "Slice Start",
    "workbench.sliceEnd": "Slice End",
    "workbench.contextSize": "Context Size",
    "workbench.stride": "Window Step",
    "workbench.patchSize": "Patch Size",
    "workbench.windowIndex": "Window Index",
    "workbench.windowRuleHint": "Uses the requested stride. Tail remainder and short samples are kept as right-padded windows.",
    "workbench.loadSamples": "Load Samples",
    "workbench.previewPackedSample": "Preview Sample Slice",
    "workbench.previewWindow": "Preview Window",
    "workbench.sampleCanvasTitle": "Packed Sample Slice",
    "workbench.windowCanvasTitle": "Training Window",
    "workbench.patchCanvasTitle": "Patch Labels",
    "workbench.trainingRail": "Training Monitor",
    "workbench.trainingTitle": "Launch Training And Monitor Progress",
    "workbench.trainingSubtitle": "Start training from the generated config, stream logs, and refresh saved metrics and quality reports in place.",
    "workbench.trainConfigPath": "Train Config Path",
    "workbench.trainOutputDir": "Train Output Dir",
    "workbench.trainDeviceOverride": "Device Override",
    "workbench.trainEpochOverride": "Epoch Override",
    "workbench.inspectData": "Run data quality inspection before training",
    "workbench.inspectMaxSamples": "Inspect Max Samples",
    "workbench.startTraining": "Start Training",
    "workbench.refreshTraining": "Refresh Metrics",
    "workbench.lossCanvasTitle": "Loss Curve",
    "workbench.metricCanvasTitle": "Quality Curves",
    "workbench.calibrationCanvasTitle": "Behavior And Calibration",
    "workbench.lossCanvasHint": "Track total, patch anomaly, point auxiliary, and reconstruction loss together.",
    "workbench.metricCanvasHint": "Watch patch-level and point-level quality instead of waiting for the final summary.",
    "workbench.calibrationCanvasHint": "Compare patch and point positive rates, threshold, and mask usage.",
    "workbench.trainingProgressEyebrow": "Live Progress",
    "workbench.trainingProgressTitle": "Current Run",
    "workbench.trainingDetailsTitle": "Raw Training Details",
    "workbench.trainingDetailsHint": "Open JSON and subprocess logs only when you need low-level debugging detail.",
    "workbench.kpi.epoch": "Epoch progress",
    "workbench.kpi.monitor": "Best monitored metric",
    "workbench.kpi.trainLoss": "Train total loss",
    "workbench.kpi.valQuality": "Latest validation signal",
    "workbench.kpi.pointHead": "Observation-space auxiliary head",
    "workbench.kpi.learningRate": "Learning rate",
    "workbench.kpi.balance": "Positive-rate gap",
    "workbench.kpi.runtime": "Elapsed / ETA",
    "workbench.kpi.threshold": "Detection threshold",
    "workbench.galleryEmpty": "Build a preview gallery to compare multiple sampled sequences.",
    "workbench.samplesEmpty": "Load a run and split to browse packed samples.",
    "workbench.trainingIdle": "Training has not started yet.",
    "workbench.runLoaded": "Run metadata loaded.",
    "workbench.galleryLoaded": "Preview gallery updated.",
    "workbench.sampleLoaded": "Packed sample preview updated.",
    "workbench.windowLoaded": "Training window preview updated.",
    "workbench.trainingStarted": "Training job started.",
    "workbench.generationStarted": "Dataset generation job started.",
    "workbench.error.missingTrainConfig": "Missing training config path.",
    "workbench.error.missingRunLookup": "Provide a run path or packed root before loading an existing run.",
    "workbench.label.seed": "seed",
    "workbench.label.events": "events",
    "workbench.label.local": "local",
    "workbench.label.seasonal": "seasonal",
    "workbench.label.length": "len",
    "workbench.label.lengthShort": "L",
    "workbench.label.seriesShort": "D",
    "workbench.label.feature": "feat",
    "workbench.label.ratio": "ratio",
    "workbench.label.windows": "windows",
    "workbench.label.window": "window",
    "workbench.label.effective": "effective",
    "workbench.label.patches": "patches",
    "workbench.label.padded": "padded",
    "workbench.label.validPatches": "valid patches",
    "workbench.label.tail": "tail",
    "workbench.label.short": "short",
    "workbench.label.job": "job",
    "workbench.label.status": "status",
    "workbench.label.started": "started",
    "workbench.label.finished": "finished",
    "toolbar.lang": "中文",
    "config.title": "Parameters",
    "config.subtitle": "Edit every config field. Small text now explains the parameter instead of repeating the raw path.",
    "preview.title": "Sample Preview",
    "preview.subtitle": "Preview runs in memory and does not write files to outputs/.",
    "tabs.series": "Series",
    "tabs.mask": "Mask",
    "tabs.dag": "DAG",
    "tabs.metadata": "Metadata",
    "preview.dataset": "Dataset",
    "preview.maxNodes": "Max Nodes",
    "preview.currentLayer": "Current Layer",
    "preview.pipelineRail": "Pipeline Inspector",
    "preview.nodes": "Node Filter",
    "preview.nodesHint": "Choose which nodes are visible in the current chart.",
    "preview.heroEyebrow": "Focused View",
    "preview.layerSummary": "Layer Summary",
    "preview.stageFlow": "Stage Flow",
    "preview.nodeLegend": "Interactive Legend",
    "preview.shadedWindows": "Shaded windows",
    "preview.resetNodes": "First N",
    "preview.allNodes": "All",
    "mask.subtitle": "Binary point mask over time for selected nodes.",
    "dag.subtitle": "Arrow direction is parent to child.",
    "metadata.summaryTitle": "Event Overview",
    "metadata.summarySubtitle": "Counts are grouped by family, propagation mode, and target component.",
    "metadata.eventsTitle": "Events",
    "metadata.eventsSubtitle": "Inspect realized event placement and sampled parameters.",
    "metadata.stagesTitle": "Structured Metadata",
    "metadata.stagesSubtitle": "Inspect the stage-scoped records generated by the current preview sample.",
    "metadata.rawTitle": "Raw Metadata",
    "metadata.rawHint": "Open the raw JSON only when you need the full debug payload.",
    "metadata.metric.events": "Event count",
    "metadata.metric.family": "Family",
    "metadata.metric.propagation": "Propagation",
    "metadata.metric.target": "Target component",
    "metadata.stage.sample": "Sample",
    "metadata.stage.stage1": "Stage 1",
    "metadata.stage.stage2": "Stage 2",
    "metadata.stage.stage3": "Stage 3",
    "metadata.field.seed": "Seed state",
    "metadata.field.records": "Param records",
    "metadata.field.mode": "Mode",
    "metadata.field.maxLag": "Max lag",
    "metadata.field.localSampled": "Local sampled",
    "metadata.field.seasonalSampled": "Seasonal sampled",
    "metadata.value.disabled": "Disabled",
    "metadata.value.arx": "ARX active",
    "import.title": "Import Config",
    "import.subtitle": "Paste a JSON or YAML config. The app validates the formal schema and fills missing fields from defaults.",
    "import.note": "Runtime and Studio import both accept only the formal schema. Missing fields are filled from defaults after validation.",
    "import.close": "Close",
    "import.apply": "Apply Imported Config",
    "import.placeholder": "{\n  \"seed\": 17\n}",
    "common.enabled": "Enabled",
    "common.yes": "Yes",
    "common.no": "No",
    "common.node": "node",
    "common.absMean": "Abs mean",
    "common.absMax": "Abs max",
    "common.length": "Length",
    "common.selectedNodes": "Selected nodes",
    "common.group": "Group",
    "common.kind": "Kind",
    "common.window": "Window",
    "common.active": "Active",
    "common.pipelineStage": "Pipeline stage",
    "status.defaultsRestored": "Defaults restored.",
    "status.randomizing": "Randomizing all parameters...",
    "status.randomized": "All parameters were filled with a valid randomized config.",
    "status.previewing": "Generating in-memory preview...",
    "status.previewed": "Preview updated.",
    "status.importing": "Importing and validating config...",
    "status.imported": "Imported config applied.",
    "status.previewEmpty": "Generate a preview to inspect events and metadata.",
    "status.layerEmpty": "Run a preview to inspect stage outputs and dataset stats.",
    "status.eventsEmpty": "No realized events for this preview.",
    "status.loadFailed": "Failed to load studio.",
    "summary.length": "Length",
    "summary.series": "Series",
    "summary.events": "Events",
    "summary.local": "Local",
    "summary.seasonal": "Seasonal",
    "summary.endogenous": "Endogenous",
    "summary.anomalous": "Anomalous",
    "stage.stage1": "Stage 1",
    "stage.stage1.detail": "Baseline built from trend, seasonality, and noise.",
    "stage.causal": "Causal",
    "stage.causal.detail": "DAG + ARX mixing delta applied to baseline outputs.",
    "stage.local": "Local",
    "stage.local.detail": "Local anomaly templates injected before and after causal propagation.",
    "stage.seasonal": "Seasonal",
    "stage.seasonal.detail": "Seasonal component operators converted into seasonal deltas.",
    "stage.final": "Final",
    "stage.final.detail": "Observed sample relative to the normal reference path.",
    "dataset.defaultDescription": "Inspect how this layer contributes to the final observed series.",
    "dataset.observed.description": "The final anomalous observation after baseline synthesis, causal mixing, and all anomaly injections.",
    "dataset.final_anomaly_delta.description": "Difference between the final observed series and the causal normal reference.",
    "dataset.stage2_normal.description": "Normal reference after causal propagation with the same baseline and graph settings.",
    "dataset.stage2_causal_effect.description": "Pure causal mixing contribution measured against the Stage 1 baseline.",
    "dataset.stage3_pre_causal_local_delta.description": "Local anomaly contribution injected before the causal simulator.",
    "dataset.stage3_post_causal_local_delta.description": "Local anomaly contribution injected directly on observed outputs.",
    "dataset.stage3_seasonal_delta.description": "Seasonal-component delta generated by seasonal anomaly operators.",
    "dataset.stage1_baseline.description": "Combined baseline signal before causal structure and anomalies are applied.",
    "dataset.stage1_trend.description": "Trend component only.",
    "dataset.stage1_seasonality.description": "Seasonality component only.",
    "dataset.stage1_noise.description": "Noise component only.",
    "event.total": "total",
    "event.local": "local",
    "event.seasonal": "seasonal",
    "event.endogenous": "endogenous",
    "event.target": "Target",
    "event.affected": "Affected",
    "event.rootCause": "Root cause",
    "event.params": "Params",
    "validation.configure": "Parameter description unavailable."
  },
  zh: {}
};

const TOKEN_LABELS = {
  en: {
    none: "None",
    sine: "Sine",
    square: "Square",
    triangle: "Triangle",
    wavelet: "Wavelet",
    increase: "Increase",
    decrease: "Decrease",
    keep_steady: "Keep Steady",
    multiple: "Multiple",
    arima: "ARIMA",
    low: "Low",
    high: "High",
    almost_none: "Almost None",
    moderate: "Moderate",
    local: "Local",
    seasonal: "Seasonal",
    observed: "Observed",
    seasonality: "Seasonality",
    baseline: "Baseline",
    endogenous: "Endogenous",
    exogenous: "Exogenous",
    morlet: "Morlet",
    ricker: "Ricker",
    haar: "Haar",
    gaus: "Gaussian",
    mexh: "Mexican Hat",
    shan: "Shannon",
    uniform: "Uniform",
    truncated_exponential: "Truncated Exponential",
    seasonal_eligible: "Seasonal Eligible",
    upward_spike: "Upward Spike",
    downward_spike: "Downward Spike",
    continuous_upward_spikes: "Continuous Upward Spikes",
    continuous_downward_spikes: "Continuous Downward Spikes",
    wide_upward_spike: "Wide Upward Spike",
    wide_downward_spike: "Wide Downward Spike",
    outlier: "Outlier",
    sudden_increase: "Sudden Increase",
    sudden_decrease: "Sudden Decrease",
    convex_plateau: "Convex Plateau",
    concave_plateau: "Concave Plateau",
    plateau: "Plateau",
    rapid_rise_slow_decline: "Rapid Rise -> Slow Decline",
    slow_rise_rapid_decline: "Slow Rise -> Rapid Decline",
    rapid_decline_slow_rise: "Rapid Decline -> Slow Rise",
    slow_decline_rapid_rise: "Slow Decline -> Rapid Rise",
    decrease_after_upward_spike: "Decrease After Upward Spike",
    increase_after_downward_spike: "Increase After Downward Spike",
    increase_after_upward_spike: "Increase After Upward Spike",
    decrease_after_downward_spike: "Decrease After Downward Spike",
    shake: "Shake",
    waveform_inversion: "Waveform Inversion",
    amplitude_scaling: "Amplitude Scaling",
    frequency_change: "Frequency Change",
    phase_shift: "Phase Shift",
    noise_injection: "Noise Injection",
    waveform_change: "Waveform Change",
    add_harmonic: "Add Harmonic",
    remove_harmonic: "Remove Harmonic",
    modify_harmonic_phase: "Modify Harmonic Phase",
    modify_modulation_depth: "Modify Modulation Depth",
    modify_modulation_frequency: "Modify Modulation Frequency",
    modify_modulation_phase: "Modify Modulation Phase",
    pulse_shift: "Pulse Shift",
    pulse_width_modulation: "Pulse Width Modulation",
    wavelet_family_change: "Wavelet Family Change",
    wavelet_scale_change: "Wavelet Scale Change",
    wavelet_shift_change: "Wavelet Shift Change",
    wavelet_amplitude_change: "Wavelet Amplitude Change",
    add_wavelet: "Add Wavelet",
    remove_wavelet: "Remove Wavelet",
  },
  zh: {
    none: "无",
    sine: "正弦",
    square: "方波",
    triangle: "三角波",
    wavelet: "小波",
    increase: "上升",
    decrease: "下降",
    keep_steady: "平稳",
    multiple: "多段",
    arima: "ARIMA",
    low: "低频",
    high: "高频",
    almost_none: "几乎无噪声",
    moderate: "中等",
    local: "局部",
    seasonal: "季节",
    observed: "观测层",
    seasonality: "季节组件",
    baseline: "基线层",
    endogenous: "内生",
    exogenous: "外生",
    morlet: "Morlet",
    ricker: "Ricker",
    haar: "Haar",
    gaus: "高斯",
    mexh: "墨西哥帽",
    shan: "Shannon",
    uniform: "均匀",
    truncated_exponential: "截断指数",
    seasonal_eligible: "季节有效节点",
    upward_spike: "上升尖峰",
    downward_spike: "下降尖峰",
    continuous_upward_spikes: "连续上升尖峰",
    continuous_downward_spikes: "连续下降尖峰",
    wide_upward_spike: "宽上升尖峰",
    wide_downward_spike: "宽下降尖峰",
    outlier: "异常点",
    sudden_increase: "突增",
    sudden_decrease: "突降",
    convex_plateau: "凸平台",
    concave_plateau: "凹平台",
    plateau: "平台",
    rapid_rise_slow_decline: "快升慢降",
    slow_rise_rapid_decline: "慢升快降",
    rapid_decline_slow_rise: "快降慢升",
    slow_decline_rapid_rise: "慢降快升",
    decrease_after_upward_spike: "上升尖峰后下降",
    increase_after_downward_spike: "下降尖峰后上升",
    increase_after_upward_spike: "上升尖峰后上升",
    decrease_after_downward_spike: "下降尖峰后下降",
    shake: "抖动",
    waveform_inversion: "波形反转",
    amplitude_scaling: "幅值缩放",
    frequency_change: "频率变化",
    phase_shift: "相位偏移",
    noise_injection: "噪声注入",
    waveform_change: "波形变换",
    add_harmonic: "添加谐波",
    remove_harmonic: "移除谐波",
    modify_harmonic_phase: "修改谐波相位",
    modify_modulation_depth: "修改调制深度",
    modify_modulation_frequency: "修改调制频率",
    modify_modulation_phase: "修改调制相位",
    pulse_shift: "脉冲平移",
    pulse_width_modulation: "脉宽调制",
    wavelet_family_change: "小波族变化",
    wavelet_scale_change: "小波尺度变化",
    wavelet_shift_change: "小波平移变化",
    wavelet_amplitude_change: "小波幅值变化",
    add_wavelet: "添加小波",
    remove_wavelet: "移除小波",
  },
};

const DATASET_LABELS = {
  en: {
    observed: "Final Observed",
    final_anomaly_delta: "Observed - Stage2 Normal",
    stage2_normal: "Stage2 Normal",
    stage2_causal_effect: "Stage2 Normal - Stage1 Baseline",
    stage3_pre_causal_local_delta: "Pre-Causal Local Delta",
    stage3_post_causal_local_delta: "Post-Causal Local Delta",
    stage3_seasonal_delta: "Seasonal Delta",
    stage1_baseline: "Stage1 Baseline",
    stage1_trend: "Stage1 Trend",
    stage1_seasonality: "Stage1 Seasonality",
    stage1_noise: "Stage1 Noise",
  },
  zh: {
    observed: "最终观测",
    final_anomaly_delta: "最终观测 - Stage2 正常参考",
    stage2_normal: "Stage2 正常参考",
    stage2_causal_effect: "Stage2 正常参考 - Stage1 基线",
    stage3_pre_causal_local_delta: "因果前局部异常增量",
    stage3_post_causal_local_delta: "因果后局部异常增量",
    stage3_seasonal_delta: "季节异常增量",
    stage1_baseline: "Stage1 基线",
    stage1_trend: "Stage1 趋势",
    stage1_seasonality: "Stage1 季节性",
    stage1_noise: "Stage1 噪声",
  },
};

const DATASET_GROUP_LABELS = {
  en: { Final: "Final", Causal: "Causal", Local: "Local", Seasonal: "Seasonal", Stage1: "Stage 1" },
  zh: { Final: "最终输出", Causal: "因果", Local: "局部异常", Seasonal: "季节异常", Stage1: "阶段 1" },
};

const DATASET_KIND_LABELS = {
  en: { signal: "Signal", delta: "Delta", component: "Component" },
  zh: { signal: "信号", delta: "增量", component: "组件" },
};

const METRIC_LABELS = {
  en: {
    total_loss: "Total loss",
    anomaly_loss: "Anomaly loss",
    point_anomaly_loss: "Point anomaly loss",
    reconstruction_loss: "Reconstruction loss",
    patch_accuracy: "Patch accuracy",
    point_accuracy: "Point accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1",
    pr_auc: "PR-AUC",
    predicted_positive_rate: "Predicted positive rate",
    target_positive_rate: "Target positive rate",
    point_predicted_positive_rate: "Point predicted positive rate",
    point_target_positive_rate: "Point target positive rate",
    anomaly_weight: "Patch anomaly weight",
    point_anomaly_weight: "Point anomaly weight",
    threshold: "Threshold",
    reconstruction_mask_fraction: "Mask fraction",
    reconstruction_used_mask: "Mask enabled",
  },
  zh: {
    total_loss: "总损失",
    anomaly_loss: "异常损失",
    point_anomaly_loss: "点级异常损失",
    reconstruction_loss: "重建损失",
    patch_accuracy: "Patch 准确率",
    point_accuracy: "点级准确率",
    precision: "精确率",
    recall: "召回率",
    f1: "F1",
    pr_auc: "PR-AUC",
    predicted_positive_rate: "预测正例占比",
    target_positive_rate: "目标正例占比",
    point_predicted_positive_rate: "点级预测正例占比",
    point_target_positive_rate: "点级目标正例占比",
    anomaly_weight: "Patch 异常权重",
    point_anomaly_weight: "点级异常权重",
    threshold: "阈值",
    reconstruction_mask_fraction: "掩码占比",
    reconstruction_used_mask: "启用掩码",
  },
};
const state = {
  defaults: null,
  config: null,
  ui: null,
  preview: null,
  locale: "en",
  selectedNodes: [],
  activeTab: "series",
  hoveredEventIndex: null,
  pinnedEventIndex: null,
  configVersion: 0,
  requestIds: {
    randomize: 0,
    preview: 0,
    importConfig: 0,
  },
  pending: {
    randomize: false,
    preview: false,
    importConfig: false,
  },
};

const dom = {
  form: document.getElementById("config-form"),
  status: document.getElementById("status-banner"),
  toggleLanguage: document.getElementById("toggle-language"),
  openImportModal: document.getElementById("open-import-modal"),
  restore: document.getElementById("restore-defaults"),
  randomize: document.getElementById("randomize-config"),
  preview: document.getElementById("preview-sample"),
  summaryChips: document.getElementById("summary-chips"),
  datasetSelect: document.getElementById("dataset-select"),
  seriesTitle: document.getElementById("series-title"),
  seriesSubtitle: document.getElementById("series-subtitle"),
  seriesBadges: document.getElementById("series-badges"),
  seriesLegend: document.getElementById("series-legend"),
  seriesFooter: document.getElementById("series-footer"),
  datasetHint: document.getElementById("dataset-hint"),
  datasetStagePill: document.getElementById("dataset-stage-pill"),
  pipelineRail: document.getElementById("pipeline-rail"),
  maxNodesSelect: document.getElementById("max-nodes-select"),
  nodeSelector: document.getElementById("node-selector"),
  resetNodeSelection: document.getElementById("reset-node-selection"),
  selectAllNodes: document.getElementById("select-all-nodes"),
  seriesCanvas: document.getElementById("series-canvas"),
  maskCanvas: document.getElementById("mask-canvas"),
  dagSvg: document.getElementById("dag-svg"),
  metadataJson: document.getElementById("metadata-json"),
  metadataStages: document.getElementById("metadata-stages"),
  eventSummary: document.getElementById("event-summary"),
  eventsTable: document.getElementById("events-table"),
  previewTabs: document.querySelector(".preview-tabs"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
  importModal: document.getElementById("import-modal"),
  closeImportModal: document.getElementById("close-import-modal"),
  importTextarea: document.getElementById("import-textarea"),
  applyImportConfig: document.getElementById("apply-import-config"),
  translatables: Array.from(document.querySelectorAll("[data-i18n]")),
};

const {
  clearCanvas,
  drawDag,
  drawLineChart,
  drawMaskHeatmap,
  drawMetricChart,
  drawPatchBarChart,
  drawSeriesPairChart,
  getNodeColor,
  palette,
  rememberCanvasLogicalSize,
} = window.TsadCharts;

const responsiveCanvasBindings = new Map();
const pendingResponsiveRenders = new Set();
let responsiveCanvasObserver = null;
let responsiveCanvasFrame = 0;
let responsiveCanvasesBound = false;

function queueResponsiveRender(renderFn) {
  if (typeof renderFn !== "function") {
    return;
  }
  pendingResponsiveRenders.add(renderFn);
  if (responsiveCanvasFrame) {
    return;
  }
  responsiveCanvasFrame = window.requestAnimationFrame(() => {
    responsiveCanvasFrame = 0;
    const renders = Array.from(pendingResponsiveRenders);
    pendingResponsiveRenders.clear();
    renders.forEach((fn) => fn());
  });
}

function bindResponsiveCanvas(canvas, renderFn, options = {}) {
  const { observe = true } = options;
  if (!canvas || typeof renderFn !== "function" || responsiveCanvasBindings.has(canvas)) {
    return;
  }
  rememberCanvasLogicalSize(canvas);
  responsiveCanvasBindings.set(canvas, renderFn);
  if (observe && responsiveCanvasObserver) {
    responsiveCanvasObserver.observe(canvas);
  }
}

function redrawResponsiveCanvases() {
  const renderFns = new Set(responsiveCanvasBindings.values());
  renderFns.forEach((fn) => queueResponsiveRender(fn));
}

function queueActivePreviewTabRender(tabId = state.activeTab) {
  const renderFn = {
    series: renderSeries,
    mask: renderMask,
    dag: renderDag,
  }[tabId];
  if (typeof renderFn !== "function") {
    return;
  }
  window.requestAnimationFrame(() => {
    queueResponsiveRender(renderFn);
  });
}

function initResponsiveCanvasRegistry() {
  if (responsiveCanvasesBound) {
    return;
  }
  responsiveCanvasesBound = true;
  if ("ResizeObserver" in window) {
    responsiveCanvasObserver = new ResizeObserver((entries) => {
      const renderFns = new Set();
      entries.forEach((entry) => {
        const renderFn = responsiveCanvasBindings.get(entry.target);
        if (renderFn) {
          renderFns.add(renderFn);
        }
      });
      renderFns.forEach((fn) => queueResponsiveRender(fn));
    });
  }
  window.addEventListener("resize", redrawResponsiveCanvases, { passive: true });
  bindResponsiveCanvas(dom.seriesCanvas, renderSeries, { observe: false });
  bindResponsiveCanvas(dom.maskCanvas, renderMask, { observe: false });
  bindResponsiveCanvas(workbenchDom.sampleCanvas, workbenchInspector.renderVisuals);
  bindResponsiveCanvas(workbenchDom.windowCanvas, workbenchInspector.renderVisuals);
  bindResponsiveCanvas(workbenchDom.patchCanvas, workbenchInspector.renderVisuals);
  bindResponsiveCanvas(workbenchDom.lossCanvas, trainingMonitor.render);
  bindResponsiveCanvas(workbenchDom.qualityCanvas, trainingMonitor.render);
  bindResponsiveCanvas(workbenchDom.calibrationCanvas, trainingMonitor.render);
  redrawResponsiveCanvases();
}

async function init() {
  const response = await fetch("/api/bootstrap");
  const payload = await response.json();
  state.defaults = payload.defaults;
  state.config = deepClone(payload.defaults);
  state.ui = payload.ui;
  state.configVersion = 1;
  bindEvents();
  applyLocale();
  setActiveTab(state.activeTab);
  renderForm();
  clearPreview();
  workbenchController.initialize(payload.workbench ?? {});
  initResponsiveCanvasRegistry();
  updateToolbarState();
}

function bindEvents() {
  dom.toggleLanguage.addEventListener("click", () => {
    state.locale = state.locale === "en" ? "zh" : "en";
    applyLocale();
    renderForm();
    if (state.preview) {
      renderPreview();
    } else {
      clearPreview();
    }
  });

  dom.openImportModal.addEventListener("click", () => {
    if (isBusy()) {
      return;
    }
    dom.importTextarea.value = JSON.stringify(state.config, null, 2);
    openImportModal();
  });

  dom.closeImportModal.addEventListener("click", closeImportModal);
  dom.importModal.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.dataset.closeModal === "true") {
      closeImportModal();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !dom.importModal.classList.contains("hidden")) {
      closeImportModal();
    }
  });

  dom.restore.addEventListener("click", () => {
    replaceConfig(deepClone(state.defaults));
    setStatus(t("status.defaultsRestored"), "ok");
  });

  dom.randomize.addEventListener("click", async () => {
    if (!flushActiveNumericInput()) {
      return;
    }
    const requestId = ++state.requestIds.randomize;
    const startedVersion = state.configVersion;
    setPending("randomize", true);
    setStatus(t("status.randomizing"), "ok");
    try {
      const response = await fetch("/api/randomize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: state.config.seed ?? undefined }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error ?? t("status.randomizing"), "error");
        return;
      }
      if (requestId !== state.requestIds.randomize || state.configVersion !== startedVersion) {
        return;
      }
      replaceConfig(payload.config);
      setStatus(t("status.randomized"), "ok");
    } catch (error) {
      if (requestId !== state.requestIds.randomize) {
        return;
      }
      setStatus(error.message ?? t("status.randomizing"), "error");
    } finally {
      if (requestId === state.requestIds.randomize) {
        setPending("randomize", false);
      }
    }
  });

  dom.preview.addEventListener("click", async () => {
    if (!flushActiveNumericInput()) {
      return;
    }
    const requestId = ++state.requestIds.preview;
    const startedVersion = state.configVersion;
    const requestConfig = deepClone(state.config);
    setPending("preview", true);
    setStatus(t("status.previewing"), "ok");
    try {
      const response = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: requestConfig }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error ?? t("status.previewing"), "error");
        return;
      }
      if (requestId !== state.requestIds.preview || state.configVersion !== startedVersion) {
        return;
      }
      state.preview = payload.preview;
      state.hoveredEventIndex = null;
      state.pinnedEventIndex = null;
      resetNodeSelection();
      renderPreview();
      setStatus(t("status.previewed"), "ok");
    } catch (error) {
      if (requestId !== state.requestIds.preview) {
        return;
      }
      setStatus(error.message ?? t("status.previewing"), "error");
    } finally {
      if (requestId === state.requestIds.preview) {
        setPending("preview", false);
      }
    }
  });

  dom.applyImportConfig.addEventListener("click", async () => {
    const requestId = ++state.requestIds.importConfig;
    const payloadText = dom.importTextarea.value.trim();
    if (!payloadText) {
      setStatus(buildValueRequiredMessage(t("import.title")), "error");
      return;
    }
    setPending("importConfig", true);
    setStatus(t("status.importing"), "ok");
    try {
      const response = await fetch("/api/import-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: payloadText }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error ?? t("status.importing"), "error");
        return;
      }
      if (requestId !== state.requestIds.importConfig) {
        return;
      }
      replaceConfig(payload.config);
      closeImportModal();
      setStatus(t("status.imported"), "ok");
    } catch (error) {
      if (requestId !== state.requestIds.importConfig) {
        return;
      }
      setStatus(error.message ?? t("status.importing"), "error");
    } finally {
      if (requestId === state.requestIds.importConfig) {
        setPending("importConfig", false);
      }
    }
  });

  dom.datasetSelect.addEventListener("change", () => {
    renderSeries();
    renderPipelineRail();
  });

  dom.resetNodeSelection.addEventListener("click", () => {
    if (!state.preview) {
      return;
    }
    resetNodeSelection();
    renderNodeSelector();
    renderSeries();
    renderMask();
    renderDag();
  });

  dom.selectAllNodes.addEventListener("click", () => {
    if (!state.preview) {
      return;
    }
    state.selectedNodes = Array.from({ length: state.preview.summary.num_series }, (_, index) => index);
    renderNodeSelector();
    renderSeries();
    renderMask();
    renderDag();
  });

  dom.maxNodesSelect.addEventListener("change", () => {
    if (!state.preview) {
      return;
    }
    resetNodeSelection();
    renderNodeSelector();
    renderSeries();
    renderMask();
    renderDag();
    updateDatasetHint(dom.datasetSelect.value);
  });

  dom.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      setActiveTab(tab.dataset.tab || "series");
    });
  });

  dom.eventsTable.addEventListener("click", (event) => {
    const card = event.target instanceof HTMLElement ? event.target.closest(".event-card[data-event-index]") : null;
    if (!card) {
      return;
    }
    togglePinnedEvent(Number(card.dataset.eventIndex));
  });

  dom.eventsTable.addEventListener("keydown", (event) => {
    if (!(event.target instanceof HTMLElement) || !["Enter", " "].includes(event.key)) {
      return;
    }
    const card = event.target.closest(".event-card[data-event-index]");
    if (!card) {
      return;
    }
    event.preventDefault();
    togglePinnedEvent(Number(card.dataset.eventIndex));
  });

  dom.eventsTable.addEventListener("mouseover", (event) => {
    if (!(event.target instanceof HTMLElement)) {
      return;
    }
    const card = event.target.closest(".event-card[data-event-index]");
    if (!card) {
      return;
    }
    setHoveredEvent(Number(card.dataset.eventIndex));
  });

  dom.eventsTable.addEventListener("mouseleave", () => {
    clearHoveredEvent();
  });

  dom.eventsTable.addEventListener("focusin", (event) => {
    if (!(event.target instanceof HTMLElement)) {
      return;
    }
    const card = event.target.closest(".event-card[data-event-index]");
    if (!card) {
      return;
    }
    setHoveredEvent(Number(card.dataset.eventIndex));
  });

  dom.eventsTable.addEventListener("focusout", (event) => {
    if (!(event.relatedTarget instanceof HTMLElement) || !dom.eventsTable.contains(event.relatedTarget)) {
      clearHoveredEvent();
    }
  });
}

function setActiveTab(tabId) {
  state.activeTab = tabId;
  dom.tabs.forEach((item) => {
    const isActive = item.dataset.tab === tabId;
    item.classList.toggle("active", isActive);
    item.setAttribute("aria-selected", isActive ? "true" : "false");
    item.tabIndex = isActive ? 0 : -1;
  });
  dom.tabPanels.forEach((panel) => {
    const isActive = panel.id === `tab-${tabId}`;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });
  queueActivePreviewTabRender(tabId);
}

function getActiveEventIndex() {
  return state.pinnedEventIndex ?? state.hoveredEventIndex;
}

function getActivePreviewEvent() {
  const activeIndex = getActiveEventIndex();
  const events = state.preview?.labels?.events ?? [];
  if (!Number.isInteger(activeIndex) || activeIndex < 0 || activeIndex >= events.length) {
    return null;
  }
  return { ...events[activeIndex], index: activeIndex };
}

function syncEventCardStates() {
  const activeIndex = getActiveEventIndex();
  const pinnedIndex = state.pinnedEventIndex;
  Array.from(dom.eventsTable.querySelectorAll(".event-card[data-event-index]")).forEach((card) => {
    const cardIndex = Number(card.dataset.eventIndex);
    card.classList.toggle("active", cardIndex === activeIndex);
    card.classList.toggle("pinned", cardIndex === pinnedIndex);
    card.setAttribute("aria-pressed", cardIndex === pinnedIndex ? "true" : "false");
  });
}

function refreshEventHighlight() {
  if (state.preview && state.activeTab === "series") {
    queueResponsiveRender(renderSeries);
  }
  syncEventCardStates();
}

function buildSelectedMaskAny(pointMask, selectedNodes) {
  if (!Array.isArray(pointMask) || !Array.isArray(selectedNodes) || selectedNodes.length === 0) {
    return [];
  }
  return pointMask.map((row) => selectedNodes.some((node) => Boolean(row?.[node])));
}

function findPositiveMaskBounds(maskAny) {
  if (!Array.isArray(maskAny) || maskAny.length === 0) {
    return null;
  }
  const start = maskAny.findIndex(Boolean);
  if (start < 0) {
    return null;
  }
  let end = maskAny.length;
  while (end > start && !maskAny[end - 1]) {
    end -= 1;
  }
  return { start, end };
}

function setHoveredEvent(index) {
  if (!Number.isInteger(index) || index < 0 || state.hoveredEventIndex === index) {
    return;
  }
  state.hoveredEventIndex = index;
  refreshEventHighlight();
}

function clearHoveredEvent() {
  if (state.hoveredEventIndex == null) {
    return;
  }
  state.hoveredEventIndex = null;
  refreshEventHighlight();
}

function togglePinnedEvent(index) {
  if (!Number.isInteger(index) || index < 0) {
    return;
  }
  state.pinnedEventIndex = state.pinnedEventIndex === index ? null : index;
  state.hoveredEventIndex = null;
  setActiveTab("series");
  refreshEventHighlight();
}

function applyLocale() {
  dom.previewTabs?.setAttribute("role", "tablist");
  dom.tabs.forEach((tab) => {
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `tab-${tab.dataset.tab}`);
  });
  dom.tabPanels.forEach((panel) => {
    panel.setAttribute("role", "tabpanel");
  });
  dom.translatables.forEach((element) => {
    if (element.id === "toggle-language") {
      return;
    }
    const key = element.dataset.i18n;
    if (key) {
      element.textContent = t(key);
    }
  });
  dom.toggleLanguage.textContent = t("toolbar.lang");
  dom.importTextarea.placeholder = t("import.placeholder");
  document.documentElement.lang = state.locale === "zh" ? "zh-CN" : "en";
  refreshWorkbenchLocale();
}

function refreshWorkbenchLocale() {
  workbenchController.refreshLocale();
  workbenchInspector.renderState();
  if (state.preview) {
    renderSeriesLegend();
  }
}

function buildOverwriteConfirmMessage(runName) {
  if (state.locale === "zh") {
    return `已启用覆盖。如果运行目录已存在，将删除其中已有内容。确认继续${runName ? `：${runName}` : ""}？`;
  }
  return `Overwrite is enabled. If this run already exists, its directory will be deleted. Continue${runName ? ` with ${runName}` : ""}?`;
}

function resetNodeSelection() {
  if (!state.preview) {
    state.selectedNodes = [];
    return;
  }
  const featureCount = state.preview.summary.num_series;
  const maxNodes = Number(dom.maxNodesSelect.value);
  state.selectedNodes = Array.from({ length: Math.min(featureCount, maxNodes) }, (_, index) => index);
}

function renderForm() {
  dom.form.innerHTML = "";
  dom.form.appendChild(renderNode(state.config, "", 0));
}

function renderNode(value, path, depth) {
  if (Array.isArray(value)) {
    return renderArrayField(value, path);
  }
  if (isPlainObject(value)) {
    const details = document.createElement("details");
    details.className = "section";
    details.dataset.path = path || "root";
    if (path.startsWith("anomaly.")) {
      details.classList.add("section-anomaly");
    }
    if (/^anomaly\.(local|seasonal)\.per_type(\.|$)/.test(path)) {
      details.classList.add("section-compact");
    }
    if (depth < 2 || path === "") {
      details.open = true;
    }

    const summary = document.createElement("summary");
    const summaryPath = path || "root";
    summary.title = summaryPath;
    summary.appendChild(makeSummaryBlock(summaryPath));
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "section-body field-grid";
    Object.entries(value).forEach(([key, child]) => {
      const childPath = path ? `${path}.${key}` : key;
      if (isLeaf(child)) {
        body.appendChild(renderLeafField(child, childPath));
      } else {
        body.appendChild(renderNode(child, childPath, depth + 1));
      }
    });
    details.appendChild(body);
    return details;
  }
  return renderLeafField(value, path);
}

function renderLeafField(value, path) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.appendChild(makeLabelBlock(path));

  if (typeof value === "boolean") {
    const checkbox = document.createElement("label");
    checkbox.className = "checkbox-pill";
    checkbox.innerHTML = `<input type="checkbox" ${value ? "checked" : ""} /> ${t("common.enabled")}`;
    checkbox.querySelector("input").addEventListener("change", (event) => {
      applyConfigChange(path, event.target.checked);
    });
    wrapper.appendChild(checkbox);
    return wrapper;
  }

  const selectOptions = state.ui.selectOptions?.[path] ?? null;
  if (Array.isArray(selectOptions) && typeof value === "string") {
    const select = document.createElement("select");
    const resolvedOptions = selectOptions.includes(value) ? selectOptions : [...selectOptions, value];
    resolvedOptions.forEach((option) => {
      const item = document.createElement("option");
      item.value = option;
      item.textContent = formatOptionLabel(path, option);
      item.selected = option === value;
      select.appendChild(item);
    });
    select.addEventListener("change", () => {
      applyConfigChange(path, select.value);
    });
    wrapper.appendChild(select);
    return wrapper;
  }

  if (typeof value === "string" || value === null) {
    const input = document.createElement("input");
    input.type = "text";
    input.value = value ?? "";
    input.dataset.path = path;
    input.addEventListener("change", () => {
      applyConfigChange(path, input.value);
    });
    wrapper.appendChild(input);
    return wrapper;
  }

  const input = document.createElement("input");
  input.type = "number";
  const numericMeta = state.ui.numericBounds[path] ?? null;
  const numericKind = numericMeta?.kind ?? (Number.isInteger(value) ? "int" : "float");
  input.value = formatNumericValue(value);
  input.step = numericKind === "int" ? "1" : "0.0001";
  input.dataset.path = path;
  input.dataset.kind = numericKind;
  if (numericMeta) {
    if (numericMeta.min != null) {
      input.min = String(numericMeta.min);
    }
    if (numericMeta.max != null) {
      input.max = String(numericMeta.max);
    }
  }
  input.addEventListener("change", () => {
    commitNumericInput(input, path, numericKind);
  });
  wrapper.appendChild(input);
  return wrapper;
}

function renderArrayField(value, path) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.appendChild(makeLabelBlock(path));

  const options = state.ui.multiSelectOptions?.[path] ?? null;
  if (!Array.isArray(options)) {
    const textarea = document.createElement("textarea");
    textarea.className = "array-textarea";
    textarea.rows = Math.max(3, Math.min(6, value.length + 1));
    textarea.value = JSON.stringify(value, null, 2);
    textarea.addEventListener("change", () => {
      try {
        const parsed = JSON.parse(textarea.value);
        if (!Array.isArray(parsed)) {
          throw new Error("Config array editor expects a JSON array.");
        }
        applyConfigChange(path, parsed);
      } catch (error) {
        setStatus(error.message, "error");
        textarea.value = JSON.stringify(getAtPath(state.config, path), null, 2);
      }
    });
    wrapper.appendChild(textarea);
    return wrapper;
  }

  const group = document.createElement("div");
  group.className = "checkbox-group";

  options.forEach((option) => {
    const chip = document.createElement("label");
    chip.className = "checkbox-pill";
    chip.title = option;
    const checked = value.includes(option) ? "checked" : "";
    chip.innerHTML = `<input type="checkbox" value="${option}" ${checked} /> ${formatOptionLabel(path, option)}`;
    chip.querySelector("input").addEventListener("change", () => {
      const current = new Set(getAtPath(state.config, path));
      if (current.has(option)) {
        current.delete(option);
      } else {
        current.add(option);
      }
      if (current.size === 0) {
        current.add(option);
        chip.querySelector("input").checked = true;
      }
      applyConfigChange(path, Array.from(current).sort());
    });
    group.appendChild(chip);
  });

  wrapper.appendChild(group);
  return wrapper;
}

function makeSummaryBlock(path) {
  const block = document.createElement("div");
  block.className = "summary-block";
  const title = document.createElement("div");
  title.className = "summary-title";
  title.textContent = getLabel(path);
  const description = document.createElement("div");
  description.className = "summary-description";
  description.textContent = getDescription(path);
  block.appendChild(title);
  block.appendChild(description);
  return block;
}

function makeLabelBlock(path) {
  const block = document.createElement("div");
  block.className = "field-header";
  block.title = path;
  const label = document.createElement("div");
  label.className = "field-label";
  label.textContent = getLabel(path);
  const description = document.createElement("div");
  description.className = "field-description";
  description.textContent = getDescription(path);
  block.appendChild(label);
  block.appendChild(description);
  return block;
}

function renderPreview() {
  if (!state.preview) {
    return;
  }
  renderDatasetCatalog();
  renderSummary();
  renderPipelineRail();
  renderNodeSelector();
  renderSeries();
  renderMask();
  renderDag();
  renderMetadata();
}

function renderSummary() {
  const summary = state.preview.summary;
  dom.summaryChips.innerHTML = "";
  [
    `${t("summary.length")} ${summary.length}`,
    `${t("summary.series")} ${summary.num_series}`,
    `${t("summary.events")} ${summary.num_events}`,
    `${t("summary.local")} ${summary.num_local_events ?? 0}`,
    `${t("summary.seasonal")} ${summary.num_seasonal_events ?? 0}`,
    `${t("summary.endogenous")} ${summary.num_endogenous_events ?? 0}`,
    `${t("summary.anomalous")} ${summary.is_anomalous_sample ? t("common.yes") : t("common.no")}`,
  ].forEach((text) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = text;
    dom.summaryChips.appendChild(chip);
  });
}

function renderDatasetCatalog() {
  if (!state.preview) {
    return;
  }
  const catalog = state.preview.series_catalog ?? [];
  const currentValue = dom.datasetSelect.value;
  dom.datasetSelect.innerHTML = "";
  catalog.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = `[${getDatasetGroupLabel(entry.group)}] ${getDatasetLabel(entry)}`;
    dom.datasetSelect.appendChild(option);
  });
  const fallback = catalog.find((entry) => entry.id === "observed")?.id ?? catalog[0]?.id ?? "";
  dom.datasetSelect.value = catalog.some((entry) => entry.id === currentValue) ? currentValue : fallback;
}

function renderPipelineRail() {
  if (!state.preview) {
    return;
  }
  const debug = state.preview.debug ?? {};
  const stats = debug.series_stats ?? {};
  const activeStage = getDatasetStage(dom.datasetSelect.value);
  const cards = [
    { stageId: "Stage1", label: t("stage.stage1"), metric: stats.stage1_baseline?.abs_mean, detail: t("stage.stage1.detail") },
    { stageId: "Causal", label: t("stage.causal"), metric: stats.stage2_causal_effect?.abs_mean, detail: t("stage.causal.detail") },
    {
      stageId: "Local",
      label: t("stage.local"),
      metric: (stats.stage3_pre_causal_local_delta?.abs_mean ?? 0) + (stats.stage3_post_causal_local_delta?.abs_mean ?? 0),
      detail: t("stage.local.detail"),
    },
    { stageId: "Seasonal", label: t("stage.seasonal"), metric: stats.stage3_seasonal_delta?.abs_mean, detail: t("stage.seasonal.detail") },
    { stageId: "Final", label: t("stage.final"), metric: stats.final_anomaly_delta?.abs_mean, detail: t("stage.final.detail") },
  ];

  dom.pipelineRail.innerHTML = "";
  cards.forEach((card) => {
    const node = document.createElement("div");
    node.className = `stage-card${card.stageId === activeStage ? " active" : ""}`;
    node.innerHTML = `
      <div class="stage-kicker">${card.stageId === activeStage ? t("common.active") : t("common.pipelineStage")}</div>
      <strong>${card.label}</strong>
      <div class="stage-metric">${t("common.absMean")} ${formatMetric(card.metric)}</div>
      <div class="stage-detail">${card.detail}</div>
    `;
    dom.pipelineRail.appendChild(node);
  });
}

function renderNodeSelector() {
  if (!state.preview) {
    return;
  }
  dom.nodeSelector.innerHTML = "";
  const featureCount = state.preview.summary.num_series;
  for (let index = 0; index < featureCount; index += 1) {
    const chip = document.createElement("label");
    chip.className = "node-pill";
    chip.innerHTML = `
      <input type="checkbox" value="${index}" ${state.selectedNodes.includes(index) ? "checked" : ""} />
      <span class="node-swatch" style="background:${getNodeColor(index)}"></span>
      <span>${t("common.node")} ${index}</span>
    `;
    chip.querySelector("input").addEventListener("change", (event) => {
      const node = Number(event.target.value);
      if (event.target.checked) {
        state.selectedNodes = [...state.selectedNodes, node].sort((a, b) => a - b);
      } else {
        state.selectedNodes = state.selectedNodes.filter((value) => value !== node);
        if (state.selectedNodes.length === 0) {
          state.selectedNodes = [node];
          event.target.checked = true;
        }
      }
      renderSeries();
      renderMask();
      renderDag();
      updateDatasetHint(dom.datasetSelect.value);
    });
    dom.nodeSelector.appendChild(chip);
  }
}

function renderSeries() {
  if (!state.preview) {
    clearCanvas(dom.seriesCanvas);
    renderLegendChips(dom.seriesLegend, []);
    renderEmptyDatasetHint();
    return;
  }
  const datasetName = dom.datasetSelect.value;
  const series = state.preview.series[datasetName];
  if (!series) {
    clearCanvas(dom.seriesCanvas);
    renderLegendChips(dom.seriesLegend, []);
    renderEmptyDatasetHint();
    return;
  }
  const selectedMaskAny = buildSelectedMaskAny(state.preview.labels?.point_mask ?? [], state.selectedNodes);
  updateDatasetHint(datasetName);
  drawLineChart(dom.seriesCanvas, series, state.selectedNodes, selectedMaskAny, {
    highlightedEvent: getActivePreviewEvent(),
  });
  renderSeriesLegend();
}

function renderMask() {
  if (!state.preview) {
    clearCanvas(dom.maskCanvas);
    return;
  }
  const pointMask = Array.isArray(state.preview.labels?.point_mask) ? state.preview.labels.point_mask : [];
  const selectedMaskAny = buildSelectedMaskAny(pointMask, state.selectedNodes);
  const focusBounds = findPositiveMaskBounds(selectedMaskAny);
  if (!focusBounds) {
    drawMaskHeatmap(dom.maskCanvas, [], state.selectedNodes, {
      emptyMessage: localizeText(
        "No positive labels are present for the current node selection.",
        "当前节点选择下没有正标签区域。",
      ),
      nodeLabel: t("common.node"),
    });
    return;
  }
  drawMaskHeatmap(
    dom.maskCanvas,
    pointMask.slice(focusBounds.start, focusBounds.end),
    state.selectedNodes,
    {
      xStart: focusBounds.start,
      xEnd: focusBounds.end,
      nodeLabel: t("common.node"),
    },
  );
}

function renderDag() {
  if (!state.preview) {
    dom.dagSvg.innerHTML = "";
    return;
  }
  drawDag(dom.dagSvg, state.preview.graph.parents, state.preview.graph.topo_order, state.selectedNodes);
}

function renderMetadata() {
  renderMetadataStages();
  renderEventSummary();
  dom.metadataJson.textContent = JSON.stringify(
    {
      metadata: state.preview.metadata,
      graph: state.preview.graph,
      debug: state.preview.debug,
      labels: {
        summary: state.preview.labels.summary,
        root_cause: state.preview.labels.root_cause,
        affected_nodes: state.preview.labels.affected_nodes,
      },
    },
    null,
    2,
  );

  const events = state.preview.labels.events ?? [];
  dom.eventsTable.innerHTML = "";
  if (events.length === 0) {
    state.hoveredEventIndex = null;
    state.pinnedEventIndex = null;
    const placeholder = document.createElement("p");
    placeholder.className = "subtle";
    placeholder.textContent = t("status.eventsEmpty");
    dom.eventsTable.appendChild(placeholder);
    return;
  }

  events.forEach((event, index) => {
    const card = document.createElement("div");
    card.className = "event-card";
    card.dataset.eventIndex = String(index);
    card.tabIndex = 0;
    card.setAttribute("role", "button");

    const header = document.createElement("div");
    header.className = "event-card-header";

    const titleBlock = document.createElement("div");
    titleBlock.className = "event-card-title";

    const title = document.createElement("strong");
    title.textContent = formatTokenLabel(event.anomaly_type);
    title.title = event.anomaly_type;
    titleBlock.appendChild(title);

    const code = document.createElement("div");
    code.className = "event-card-code";
    code.textContent = event.anomaly_type;
    code.title = event.anomaly_type;
    titleBlock.appendChild(code);
    header.appendChild(titleBlock);

    const locationBadge = document.createElement("div");
    locationBadge.className = "event-card-location";
    locationBadge.textContent = `${t("common.node")} ${event.node} · [${event.t_start}, ${event.t_end})`;
    header.appendChild(locationBadge);
    card.appendChild(header);

    const row = document.createElement("div");
    row.className = "event-card-row";
    [formatTokenLabel(event.family), formatTokenLabel(event.target_component), event.is_endogenous ? formatTokenLabel("endogenous") : formatTokenLabel("exogenous")].forEach(
      (value) => {
        const pill = document.createElement("span");
        pill.className = "event-pill";
        pill.textContent = value;
        row.appendChild(pill);
      },
    );
    card.appendChild(row);

    const metaGrid = document.createElement("div");
    metaGrid.className = "event-meta-grid";
    metaGrid.appendChild(makeEventMetaItem(t("common.window"), `[${event.t_start}, ${event.t_end})`));
    metaGrid.appendChild(makeEventMetaItem(t("event.target"), formatTokenLabel(event.target_component)));
    metaGrid.appendChild(makeEventMetaItem(t("event.affected"), (event.affected_nodes ?? []).join(", ") || "-"));
    metaGrid.appendChild(makeEventMetaItem(t("event.rootCause"), event.root_cause_node ?? "-"));
    card.appendChild(metaGrid);

    const paramsLabel = document.createElement("div");
    paramsLabel.className = "event-params-label";
    paramsLabel.textContent = t("event.params");
    card.appendChild(paramsLabel);

    const params = document.createElement("pre");
    params.className = "event-param-box";
    params.textContent = JSON.stringify(event.params ?? {}, null, 2);
    card.appendChild(params);
    dom.eventsTable.appendChild(card);
  });
  syncEventCardStates();
}

function renderMetadataStages() {
  dom.metadataStages.innerHTML = "";
  if (!state.preview) {
    return;
  }

  const metadata = state.preview.metadata ?? {};
  const stage1Params = Array.isArray(metadata.stage1?.params) ? metadata.stage1.params : [];
  const stage2Params = metadata.stage2?.params ?? {};
  const stage3Sampled = metadata.stage3?.sampled_events ?? {};
  const cards = [
    {
      title: t("metadata.stage.sample"),
      items: [{ label: t("metadata.field.seed"), value: metadata.sample?.seed_state ?? "-" }],
    },
    {
      title: t("metadata.stage.stage1"),
      items: [{ label: t("metadata.field.records"), value: stage1Params.length }],
    },
    {
      title: t("metadata.stage.stage2"),
      items: [
        {
          label: t("metadata.field.mode"),
          value: stage2Params.disabled === true ? t("metadata.value.disabled") : t("metadata.value.arx"),
        },
        {
          label: t("metadata.field.maxLag"),
          value: stage2Params.disabled === true ? "-" : stage2Params.max_lag ?? "-",
        },
      ],
    },
    {
      title: t("metadata.stage.stage3"),
      items: [
        { label: t("metadata.field.localSampled"), value: Array.isArray(stage3Sampled.local) ? stage3Sampled.local.length : 0 },
        {
          label: t("metadata.field.seasonalSampled"),
          value: Array.isArray(stage3Sampled.seasonal) ? stage3Sampled.seasonal.length : 0,
        },
      ],
    },
  ];

  cards.forEach((entry) => {
    const card = document.createElement("section");
    card.className = "metadata-stage-card";

    const title = document.createElement("div");
    title.className = "metadata-stage-title";
    title.textContent = entry.title;
    card.appendChild(title);

    const items = document.createElement("div");
    items.className = "metadata-stage-items";

    entry.items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "metadata-stage-item";

      const label = document.createElement("div");
      label.className = "metadata-stage-label";
      label.textContent = item.label;
      row.appendChild(label);

      const value = document.createElement("div");
      value.className = "metadata-stage-value";
      value.textContent = String(item.value);
      row.appendChild(value);

      items.appendChild(row);
    });

    card.appendChild(items);
    dom.metadataStages.appendChild(card);
  });
}

function renderEventSummary() {
  if (!state.preview) {
    return;
  }
  const summary = state.preview.labels.summary ?? {};
  const components = summary.target_components ?? {};
  dom.eventSummary.innerHTML = "";
  const cards = [
    { label: t("event.total"), value: summary.total ?? 0, note: t("metadata.metric.events"), tone: "total" },
    { label: t("event.local"), value: summary.local ?? 0, note: t("metadata.metric.family"), tone: "local" },
    { label: t("event.seasonal"), value: summary.seasonal ?? 0, note: t("metadata.metric.family"), tone: "seasonal" },
    { label: t("event.endogenous"), value: summary.endogenous ?? 0, note: t("metadata.metric.propagation"), tone: "endogenous" },
    ...Object.entries(components).map(([key, value]) => ({
      label: formatTokenLabel(key),
      value,
      note: t("metadata.metric.target"),
      tone: "component",
    })),
  ];

  cards.forEach((item) => {
    const card = document.createElement("div");
    card.className = `event-summary-card tone-${item.tone}`;

    const note = document.createElement("div");
    note.className = "event-summary-note";
    note.textContent = item.note;
    card.appendChild(note);

    const value = document.createElement("div");
    value.className = "event-summary-value";
    value.textContent = String(item.value);
    card.appendChild(value);

    const label = document.createElement("div");
    label.className = "event-summary-label";
    label.textContent = item.label;
    card.appendChild(label);

    dom.eventSummary.appendChild(card);
  });
}

function makeEventMetaItem(labelText, valueText) {
  const item = document.createElement("div");
  item.className = "event-meta-item";

  const label = document.createElement("div");
  label.className = "event-meta-label";
  label.textContent = labelText;
  item.appendChild(label);

  const value = document.createElement("div");
  value.className = "event-meta-value";
  value.textContent = String(valueText);
  item.appendChild(value);

  return item;
}

function updateDatasetHint(datasetName) {
  const entry = (state.preview?.series_catalog ?? []).find((item) => item.id === datasetName);
  const stats = state.preview?.debug?.series_stats?.[datasetName];
  const activeEvent = getActivePreviewEvent();
  const selectedMaskAny = buildSelectedMaskAny(state.preview?.labels?.point_mask ?? [], state.selectedNodes);
  if (!entry) {
    renderEmptyDatasetHint();
    return;
  }

  dom.seriesTitle.textContent = getDatasetLabel(entry);
  dom.seriesSubtitle.textContent = getDatasetDescription(entry.id);
  dom.datasetStagePill.textContent = getDatasetGroupLabel(entry.group);
  dom.seriesBadges.innerHTML = [
    `<span class="hero-badge"><strong>${escapeHtml(t("common.group"))}</strong>${escapeHtml(getDatasetGroupLabel(entry.group))}</span>`,
    `<span class="hero-badge"><strong>${escapeHtml(t("common.kind"))}</strong>${escapeHtml(getDatasetKindLabel(entry.kind))}</span>`,
    `<span class="hero-badge"><strong>${escapeHtml(t("common.absMean"))}</strong>${escapeHtml(formatMetric(stats?.abs_mean))}</span>`,
    `<span class="hero-badge"><strong>${escapeHtml(t("common.absMax"))}</strong>${escapeHtml(formatMetric(stats?.abs_max))}</span>`,
  ].join("");

  const windowCount = countMaskWindows(selectedMaskAny);
  const footerItems = [
    `<span class="footer-chip">${escapeHtml(t("preview.shadedWindows"))}: ${escapeHtml(String(windowCount))}</span>`,
    `<span class="footer-chip">${escapeHtml(t("common.selectedNodes"))}: ${escapeHtml(String(state.selectedNodes.length))}</span>`,
    `<span class="footer-chip">${escapeHtml(t("common.length"))}: ${escapeHtml(String(state.preview.summary.length))}</span>`,
  ];
  if (activeEvent) {
    footerItems.unshift(
      `<span class="footer-chip"><strong>${escapeHtml(formatTokenLabel(activeEvent.anomaly_type))}</strong> ${escapeHtml(`[${activeEvent.t_start}, ${activeEvent.t_end})`)}</span>`,
    );
  }
  dom.seriesFooter.innerHTML = footerItems.join("");

  dom.datasetHint.innerHTML = `
    <div class="dataset-header">
      <div class="dataset-title-row">
        <div class="dataset-title">${escapeHtml(getDatasetLabel(entry))}</div>
        <div class="dataset-meta">
          <span class="event-pill">${escapeHtml(getDatasetGroupLabel(entry.group))}</span>
          <span class="event-pill">${escapeHtml(getDatasetKindLabel(entry.kind))}</span>
        </div>
      </div>
      <div class="dataset-description">${escapeHtml(getDatasetDescription(entry.id))}</div>
    </div>
    <div class="dataset-stat-grid">
      <div class="dataset-stat">
        <span class="dataset-stat-label">${escapeHtml(t("common.absMean"))}</span>
        <strong class="dataset-stat-value">${escapeHtml(formatMetric(stats?.abs_mean))}</strong>
      </div>
      <div class="dataset-stat">
        <span class="dataset-stat-label">${escapeHtml(t("common.absMax"))}</span>
        <strong class="dataset-stat-value">${escapeHtml(formatMetric(stats?.abs_max))}</strong>
      </div>
      <div class="dataset-stat">
        <span class="dataset-stat-label">${escapeHtml(t("common.length"))}</span>
        <strong class="dataset-stat-value">${escapeHtml(String(state.preview.summary.length))}</strong>
      </div>
      <div class="dataset-stat">
        <span class="dataset-stat-label">${escapeHtml(t("common.selectedNodes"))}</span>
        <strong class="dataset-stat-value">${escapeHtml(String(state.selectedNodes.length))}</strong>
      </div>
    </div>
  `;
}

function renderEmptyDatasetHint() {
  dom.seriesTitle.textContent = t("preview.title");
  dom.seriesSubtitle.textContent = t("status.layerEmpty");
  dom.seriesBadges.innerHTML = "";
  dom.seriesLegend.innerHTML = "";
  dom.seriesFooter.innerHTML = "";
  dom.datasetStagePill.textContent = "";
  dom.datasetHint.innerHTML = `<div class="dataset-empty">${escapeHtml(t("status.layerEmpty"))}</div>`;
}

function renderSeriesLegend() {
  if (!state.preview || !state.selectedNodes.length) {
    renderLegendChips(dom.seriesLegend, []);
    return;
  }
  renderLegendChips(
    dom.seriesLegend,
    state.selectedNodes.map((node) => ({
      color: getNodeColor(node),
      label: `${t("common.node")} ${node}`,
    })),
  );
}

function renderLegendChips(container, items) {
  if (!container) {
    return;
  }
  const safeItems = Array.isArray(items) ? items : [];
  container.innerHTML = safeItems
    .map(
      (item) => `
        <span class="legend-chip">
          <span class="legend-chip-swatch" style="background:${escapeHtml(String(item.color ?? "#b55232"))}"></span>
          <span>${escapeHtml(String(item.label ?? ""))}</span>
        </span>
      `,
    )
    .join("");
}

function clearPreview() {
  state.preview = null;
  state.selectedNodes = [];
  state.hoveredEventIndex = null;
  state.pinnedEventIndex = null;
  dom.summaryChips.innerHTML = "";
  dom.datasetSelect.innerHTML = "";
  renderEmptyDatasetHint();
  dom.pipelineRail.innerHTML = "";
  dom.nodeSelector.innerHTML = "";
  clearCanvas(dom.seriesCanvas);
  clearCanvas(dom.maskCanvas);
  dom.dagSvg.innerHTML = "";
  dom.metadataJson.textContent = "";
  dom.metadataStages.innerHTML = `<p class="subtle">${escapeHtml(t("status.previewEmpty"))}</p>`;
  dom.eventSummary.innerHTML = "";
  dom.eventsTable.innerHTML = `<p class="subtle">${escapeHtml(t("status.previewEmpty"))}</p>`;
  updateToolbarState();
}

function replaceConfig(nextConfig) {
  state.config = nextConfig;
  state.configVersion += 1;
  renderForm();
  clearPreview();
  hideStatus();
}

function applyConfigChange(path, value) {
  setAtPath(state.config, path, value);
  state.configVersion += 1;
  clearPreview();
  hideStatus();
}

function commitNumericInput(input, path, kind) {
  const rawValue = input.value.trim();
  if (rawValue === "") {
    input.value = formatNumericValue(getAtPath(state.config, path));
    setStatus(buildValueRequiredMessage(getLabel(path)), "error");
    return false;
  }
  const nextValue = Number(rawValue);
  const error = validateNumericValue(path, nextValue, kind);
  if (error) {
    input.value = formatNumericValue(getAtPath(state.config, path));
    setStatus(error, "error");
    return false;
  }

  const currentValue = getAtPath(state.config, path);
  input.value = formatNumericValue(nextValue);
  if (Object.is(currentValue, nextValue)) {
    clearErrorStatus();
    return true;
  }
  applyConfigChange(path, nextValue);
  return true;
}

function validateNumericValue(path, value, kind) {
  const label = getLabel(path);
  if (!Number.isFinite(value)) {
    return localizeText(`Invalid numeric value for ${label}.`, `${label} 的数值无效。`);
  }
  if (kind === "int" && !Number.isInteger(value)) {
    return localizeText(`Expected an integer for ${label}.`, `${label} 需要输入整数。`);
  }

  const numericMeta = state.ui.numericBounds[path];
  if (numericMeta) {
    const hasMin = numericMeta.min != null;
    const hasMax = numericMeta.max != null;
    if ((hasMin && value < numericMeta.min) || (hasMax && value > numericMeta.max)) {
      if (hasMin && hasMax) {
        return localizeText(
          `${label} must be between ${numericMeta.min} and ${numericMeta.max}.`,
          `${label} 必须在 ${numericMeta.min} 到 ${numericMeta.max} 之间。`,
        );
      }
      if (hasMin) {
        return localizeText(
          `${label} must be at least ${numericMeta.min}.`,
          `${label} 必须大于或等于 ${numericMeta.min}。`,
        );
      }
      return localizeText(
        `${label} must be at most ${numericMeta.max}.`,
        `${label} 必须小于或等于 ${numericMeta.max}。`,
      );
    }
  }

  if (path.endsWith(".min")) {
    const basePath = path.slice(0, -4);
    const baseLabel = getLabel(basePath);
    if (value > getAtPath(state.config, `${basePath}.max`)) {
      return localizeText(`Min cannot exceed max for ${baseLabel}.`, `${baseLabel} 的最小值不能大于最大值。`);
    }
  }

  if (path.endsWith(".max")) {
    const basePath = path.slice(0, -4);
    const baseLabel = getLabel(basePath);
    if (value < getAtPath(state.config, `${basePath}.min`)) {
      return localizeText(`Max cannot be smaller than min for ${baseLabel}.`, `${baseLabel} 的最大值不能小于最小值。`);
    }
  }
  return null;
}

function flushActiveNumericInput() {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLInputElement) || activeElement.type !== "number") {
    return true;
  }
  const path = activeElement.dataset.path;
  const kind = activeElement.dataset.kind ?? "float";
  return path ? commitNumericInput(activeElement, path, kind) : true;
}

function formatNumericValue(value) {
  return String(value);
}

function formatMetric(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(4);
}

function countMaskWindows(maskAny) {
  let windows = 0;
  let inWindow = false;
  maskAny.forEach((value) => {
    if (value && !inWindow) {
      windows += 1;
      inWindow = true;
    } else if (!value) {
      inWindow = false;
    }
  });
  return windows;
}

function setPending(action, value) {
  state.pending[action] = value;
  updateToolbarState();
}

function updateToolbarState() {
  const busy = isBusy();
  dom.restore.disabled = busy;
  dom.randomize.disabled = busy;
  dom.preview.disabled = busy;
  dom.openImportModal.disabled = busy;
  dom.applyImportConfig.disabled = state.pending.importConfig;
  if (dom.resetNodeSelection) {
    dom.resetNodeSelection.disabled = busy || !state.preview;
  }
  if (dom.selectAllNodes) {
    dom.selectAllNodes.disabled = busy || !state.preview;
  }
}

function isBusy() {
  return state.pending.randomize || state.pending.preview || state.pending.importConfig;
}

function setStatus(message, kind) {
  dom.status.textContent = message;
  dom.status.classList.remove("hidden", "error", "ok");
  dom.status.classList.add(kind);
}

function hideStatus() {
  dom.status.textContent = "";
  dom.status.classList.add("hidden");
  dom.status.classList.remove("error", "ok");
}

function clearErrorStatus() {
  if (dom.status.classList.contains("error")) {
    hideStatus();
  }
}

function getLabel(path) {
  const localeUi = getLocaleUi();
  if (!path) {
    return localeUi.pathLabels.root ?? "Root";
  }
  return localeUi.pathLabels[path] ?? path.split(".").slice(-1)[0];
}

function getDescription(path) {
  const localeUi = getLocaleUi();
  return localeUi.pathDescriptions[path] ?? describeCurrentValue(path);
}

function getLocaleUi() {
  if (!state.ui) {
    return { pathLabels: {}, pathDescriptions: {} };
  }
  return state.ui.locales?.[state.locale] ?? { pathLabels: state.ui.pathLabels ?? {}, pathDescriptions: state.ui.pathDescriptions ?? {} };
}

function describeCurrentValue(path) {
  const label = getLabel(path);
  if (!state.config) {
    return t("validation.configure");
  }

  let value;
  try {
    value = path ? getAtPath(state.config, path) : state.config;
  } catch {
    return label;
  }

  if (Array.isArray(value)) {
    if (value.length > 0 && value.every((item) => typeof item === "string")) {
      return localizeText(`Selectable list for ${label}.`, `${label} 的可选列表。`);
    }
    return localizeText(`List setting for ${label}.`, `${label} 的列表设置。`);
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    if (keys.length === 2 && keys.includes("min") && keys.includes("max")) {
      return localizeText(`Numeric range for ${label}.`, `${label} 的数值范围。`);
    }
    if (path.includes(".per_type.")) {
      return localizeText(`Per-type settings for ${label}.`, `${label} 的逐类型设置。`);
    }
    return localizeText(`Parameter group for ${label}.`, `${label} 的参数组。`);
  }
  if (typeof value === "boolean") {
    return localizeText(`Toggle for ${label}.`, `${label} 的开关。`);
  }
  if (typeof value === "number") {
    return localizeText(`Numeric setting for ${label}.`, `${label} 的数值设置。`);
  }
  if (typeof value === "string") {
    return localizeText(`Categorical option for ${label}.`, `${label} 的类别选项。`);
  }
  return label;
}

function getDatasetLabel(entry) {
  return DATASET_LABELS[state.locale]?.[entry.id] ?? entry.label;
}

function getDatasetDescription(datasetId) {
  return LOCALE_TEXT[state.locale][`dataset.${datasetId}.description`] ?? t("dataset.defaultDescription");
}

function getDatasetGroupLabel(group) {
  return DATASET_GROUP_LABELS[state.locale]?.[group] ?? group;
}

function getDatasetKindLabel(kind) {
  return DATASET_KIND_LABELS[state.locale]?.[kind] ?? kind;
}

function getDatasetStage(datasetName) {
  const entry = (state.preview?.series_catalog ?? []).find((item) => item.id === datasetName);
  return entry?.group ?? "Final";
}

function formatOptionLabel(path, option) {
  if (path === "stage1.seasonality.wavelet.contrastive.params") {
    return getLabel(`stage1.seasonality.${option}`);
  }
  return formatTokenLabel(option);
}

function formatTokenLabel(value) {
  if (value == null || value === "") {
    return "-";
  }
  return TOKEN_LABELS[state.locale]?.[value] ?? formatCodeLabel(value);
}

function formatCodeLabel(value) {
  if (value == null || value === "") {
    return "-";
  }
  const direct = TOKEN_LABELS[state.locale]?.[value];
  if (direct) {
    return direct;
  }
  return value
    .split("_")
    .map((part) => (part.length > 0 ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function localizeText(enText, zhText) {
  return state.locale === "zh" ? zhText : enText;
}

function buildValueRequiredMessage(label) {
  return localizeText(`Value required for ${label}.`, `${label} 不能为空。`);
}

function t(key) {
  return LOCALE_TEXT[state.locale]?.[key] ?? LOCALE_TEXT.en[key] ?? key;
}

function openImportModal() {
  dom.importModal.classList.remove("hidden");
  dom.importModal.setAttribute("aria-hidden", "false");
  window.setTimeout(() => dom.importTextarea.focus(), 0);
}

function closeImportModal() {
  if (state.pending.importConfig) {
    return;
  }
  dom.importModal.classList.add("hidden");
  dom.importModal.setAttribute("aria-hidden", "true");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isLeaf(value) {
  return Array.isArray(value) || !isPlainObject(value);
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function setAtPath(target, path, value) {
  const parts = path.split(".");
  let current = target;
  for (let index = 0; index < parts.length - 1; index += 1) {
    current = current[parts[index]];
  }
  current[parts[parts.length - 1]] = value;
}

function getAtPath(target, path) {
  return path.split(".").reduce((current, key) => current[key], target);
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

const zhOverrides = {
  "toolbar.lang": "English",
  "toolbar.import": "导入配置",
  "toolbar.restore": "恢复默认",
  "toolbar.randomize": "随机填充",
  "toolbar.preview": "预览样本",
  "toolbar.previewBatch": "预览画廊",
  "config.title": "参数配置",
  "config.subtitle": "可编辑全部配置字段，并显示每个参数的解释。",
  "preview.title": "样本预览",
  "preview.subtitle": "预览只在内存中运行，不写入 outputs/。",
  "tabs.series": "序列",
  "tabs.mask": "掩码",
  "tabs.dag": "DAG",
  "tabs.metadata": "元数据",
  "preview.dataset": "数据层",
  "preview.maxNodes": "最多节点",
  "preview.currentLayer": "当前层",
  "preview.pipelineRail": "流水线检查",
  "preview.nodes": "节点筛选",
  "preview.nodesHint": "选择当前图中要显示的节点。",
  "preview.heroEyebrow": "聚焦视图",
  "preview.layerSummary": "层摘要",
  "preview.stageFlow": "阶段流程",
  "preview.nodeLegend": "交互图例",
  "preview.resetNodes": "前 N 个",
  "preview.allNodes": "全部",
  "mask.subtitle": "展示所选节点的逐时刻二值点标签。",
  "dag.subtitle": "箭头方向为父节点到子节点。",
  "metadata.summaryTitle": "事件总览",
  "metadata.summarySubtitle": "按家族、传播模式和目标组件统计。",
  "metadata.eventsTitle": "事件明细",
  "metadata.eventsSubtitle": "查看本次预览的事件位置与采样参数。",
  "metadata.stagesTitle": "结构化元数据",
  "metadata.stagesSubtitle": "查看当前样本各阶段生成的元信息。",
  "metadata.rawTitle": "原始元数据",
  "metadata.rawHint": "仅在排障时展开完整 JSON。",
  "import.title": "导入配置",
  "import.subtitle": "粘贴 JSON 或 YAML，系统会校验并补齐默认字段。",
  "import.note": "运行时与 Studio 导入均使用正式 schema，缺失字段会在校验后补默认值。",
  "import.close": "关闭",
  "import.apply": "应用导入配置",
  "brand.title": "TSAD 工作台",
  "brand.subtitle": "在同一页面完成全配置编辑、多样本预览、数据集生成检查与训练监控。",
  "workbench.previewRail": "预览流程",
  "workbench.previewGalleryTitle": "预览画廊",
  "workbench.previewGallerySubtitle": "基于当前配置一次生成多个内存样本，再选择任一条加载到主预览区。",
  "workbench.previewCount": "预览数量",
  "workbench.previewSeed": "基础种子",
  "workbench.previewGenerate": "生成预览画廊",
  "workbench.generationRail": "数据流程",
  "workbench.generateTitle": "生成并打包数据集",
  "workbench.generateSubtitle": "将当前配置写为运行配置，并行生成 train/val/test 三个 split，随后完成打包并产出训练配置。",
  "workbench.runName": "运行名称",
  "workbench.trainSamples": "训练样本数",
  "workbench.valSamples": "验证样本数",
  "workbench.testSamples": "测试样本数",
  "workbench.shardSize": "每分片样本数",
  "workbench.seedBase": "基础种子",
  "workbench.trainTemplate": "训练模板",
  "workbench.trainDevice": "训练设备",
  "workbench.trainEpochs": "训练轮数",
  "workbench.trainBatchSize": "训练批大小",
  "workbench.evalBatchSize": "评估批大小",
  "workbench.overwriteRun": "运行目录存在时覆盖",
  "workbench.generateAction": "生成并打包",
  "workbench.loadRun": "加载已有运行",
  "workbench.datasetRail": "数据集检查",
  "workbench.datasetTitle": "样本与窗口查看",
  "workbench.datasetSubtitle": "浏览 packed 样本、查看序列切片，并可视化训练窗口和 patch 标签。",
  "workbench.runPath": "运行目录或 packed 根目录",
  "workbench.split": "切分",
  "workbench.sample": "样本",
  "workbench.feature": "特征",
  "workbench.sliceStart": "切片起点",
  "workbench.sliceEnd": "切片终点",
  "workbench.contextSize": "上下文长度",
  "workbench.stride": "窗口步长",
  "workbench.patchSize": "Patch 大小",
  "workbench.windowIndex": "窗口索引",
  "workbench.windowRuleHint": "按当前窗口步长切窗；尾部不足一窗和短样本都会保留，并在右侧补齐到 context_size。",
  "workbench.loadSamples": "加载样本",
  "workbench.previewPackedSample": "预览样本切片",
  "workbench.previewWindow": "预览窗口",
  "workbench.sampleCanvasTitle": "样本切片",
  "workbench.windowCanvasTitle": "训练窗口",
  "workbench.patchCanvasTitle": "Patch 标签",
  "workbench.trainingRail": "训练监控",
  "workbench.trainingTitle": "启动训练并监控进度",
  "workbench.trainingSubtitle": "基于训练配置启动训练，实时查看日志并刷新历史指标和数据体检结果。",
  "workbench.trainConfigPath": "训练配置路径",
  "workbench.trainOutputDir": "训练输出目录",
  "workbench.trainDeviceOverride": "设备覆盖",
  "workbench.trainEpochOverride": "轮数覆盖",
  "workbench.inspectData": "训练前执行数据质量体检",
  "workbench.inspectMaxSamples": "体检样本上限",
  "workbench.startTraining": "开始训练",
  "workbench.refreshTraining": "刷新指标",
  "workbench.lossCanvasTitle": "损失曲线",
  "workbench.metricCanvasTitle": "质量曲线",
  "workbench.calibrationCanvasTitle": "行为与校准",
  "workbench.lossCanvasHint": "把总损失、patch 异常损失、点级辅助损失和重建损失放在一起跟踪。",
  "workbench.metricCanvasHint": "不用等最终总结，训练中就能同时看到 patch 级和点级质量变化。",
  "workbench.calibrationCanvasHint": "同时对比 patch / 点级正例占比、阈值和掩码使用情况。",
  "workbench.trainingProgressEyebrow": "实时进度",
  "workbench.trainingProgressTitle": "当前运行",
  "workbench.trainingDetailsTitle": "训练原始明细",
  "workbench.trainingDetailsHint": "只有在需要排查底层问题时，再展开 JSON 和子进程日志。",
  "workbench.kpi.epoch": "轮次进度",
  "workbench.kpi.monitor": "最佳监控指标",
  "workbench.kpi.trainLoss": "训练总损失",
  "workbench.kpi.valQuality": "最新验证信号",
  "workbench.kpi.pointHead": "观测空间辅助头",
  "workbench.kpi.learningRate": "学习率",
  "workbench.kpi.balance": "正例占比偏差",
  "workbench.kpi.runtime": "已耗时 / 预计剩余",
  "workbench.kpi.threshold": "检测阈值",
  "workbench.galleryEmpty": "先生成预览画廊，再比较多组采样结果。",
  "workbench.samplesEmpty": "先加载运行目录和切分，再浏览 packed 样本。",
  "workbench.trainingIdle": "训练尚未开始。",
  "workbench.runLoaded": "运行信息已加载。",
  "workbench.galleryLoaded": "预览画廊已更新。",
  "workbench.sampleLoaded": "样本切片预览已更新。",
  "workbench.windowLoaded": "训练窗口预览已更新。",
  "workbench.trainingStarted": "训练任务已启动。",
  "workbench.generationStarted": "数据生成任务已启动。",
  "workbench.error.missingTrainConfig": "\u7f3a\u5c11\u8bad\u7ec3\u914d\u7f6e\u8def\u5f84\u3002",
  "workbench.error.missingRunLookup": "请先填写运行目录或 packed 根目录，再加载已有运行。",
  "workbench.label.seed": "\u79cd\u5b50",
  "workbench.label.events": "\u4e8b\u4ef6",
  "workbench.label.local": "\u5c40\u90e8",
  "workbench.label.seasonal": "\u5b63\u8282\u6027",
  "workbench.label.length": "\u957f\u5ea6",
  "workbench.label.lengthShort": "\u957f",
  "workbench.label.seriesShort": "\u7ef4",
  "workbench.label.feature": "\u7279\u5f81",
  "workbench.label.ratio": "\u5360\u6bd4",
  "workbench.label.windows": "\u7a97\u53e3",
  "workbench.label.window": "\u7a97\u53e3",
  "workbench.label.effective": "\u6709\u6548\u957f\u5ea6",
  "workbench.label.patches": "Patch \u6570",
  "workbench.label.padded": "\u8865\u9f50",
  "workbench.label.validPatches": "\u6709\u6548 Patch",
  "workbench.label.tail": "\u5c3e\u7a97",
  "workbench.label.short": "\u77ed\u6837\u672c",
  "workbench.label.job": "\u4efb\u52a1",
  "workbench.label.status": "\u72b6\u6001",
  "workbench.label.started": "\u5f00\u59cb",
  "workbench.label.finished": "\u7ed3\u675f"
};

Object.assign(LOCALE_TEXT.zh, LOCALE_TEXT.en, zhOverrides);
Object.assign(LOCALE_TEXT.zh, {
  "preview.shadedWindows": "阴影窗口",
  "metadata.metric.events": "事件数",
  "metadata.metric.family": "异常家族",
  "metadata.metric.propagation": "传播方式",
  "metadata.metric.target": "目标组件",
  "metadata.stage.sample": "样本",
  "metadata.stage.stage1": "阶段 1",
  "metadata.stage.stage2": "阶段 2",
  "metadata.stage.stage3": "阶段 3",
  "metadata.field.seed": "种子状态",
  "metadata.field.records": "参数记录数",
  "metadata.field.mode": "模式",
  "metadata.field.maxLag": "最大滞后",
  "metadata.field.localSampled": "局部事件采样数",
  "metadata.field.seasonalSampled": "季节事件采样数",
  "metadata.value.disabled": "已禁用",
  "metadata.value.arx": "ARX 已启用",
  "common.enabled": "启用",
  "common.yes": "是",
  "common.no": "否",
  "common.node": "节点",
  "common.absMean": "绝对均值",
  "common.absMax": "绝对最大值",
  "common.length": "长度",
  "common.selectedNodes": "已选节点",
  "common.group": "分组",
  "common.kind": "类型",
  "common.window": "窗口",
  "common.active": "当前阶段",
  "common.pipelineStage": "流程阶段",
  "summary.length": "长度",
  "summary.series": "序列数",
  "summary.events": "事件数",
  "summary.local": "局部异常",
  "summary.seasonal": "季节异常",
  "summary.endogenous": "内生",
  "summary.anomalous": "是否异常",
  "stage.stage1": "阶段 1",
  "stage.stage1.detail": "由趋势、季节性和噪声共同构成基线序列。",
  "stage.causal": "因果",
  "stage.causal.detail": "基于 DAG 和 ARX 把跨节点影响混入基线输出。",
  "stage.local": "局部异常",
  "stage.local.detail": "局部异常模板会在因果传播前后两个位置注入。",
  "stage.seasonal": "季节异常",
  "stage.seasonal.detail": "季节组件异常会先转换成季节增量，再叠加回观测序列。",
  "stage.final": "最终输出",
  "stage.final.detail": "最终观测结果相对于正常参考路径的偏移。",
  "event.total": "总数",
  "event.local": "局部",
  "event.seasonal": "季节",
  "event.endogenous": "内生",
  "event.target": "目标组件",
  "event.affected": "受影响节点",
  "event.rootCause": "根因节点",
  "event.params": "参数",
});

import { buildQuery, fetchJson, postJson, safeJson } from "./workbench/api.js";
import { createWorkbenchController } from "./workbench/controller.js";
import { createWorkbenchInspector } from "./workbench/inspector.js";
import { createWorkbenchDom, createWorkbenchState } from "./workbench/runtime.js";
import { createTrainingMonitor } from "./workbench/training_monitor.js";

const workbenchDom = createWorkbenchDom();
const workbenchState = createWorkbenchState();
const workbenchInspector = createWorkbenchInspector({
  dom: workbenchDom,
  state: workbenchState,
  clearCanvas,
  drawPatchBarChart,
  drawSeriesPairChart,
  escapeHtml,
  formatMetric,
  localizeText,
  t,
});
const trainingMonitor = createTrainingMonitor({
  dom: workbenchDom,
  state: workbenchState,
  clearCanvas,
  drawMetricChart,
  escapeHtml,
  formatMetric,
  getLocale: () => state.locale,
  localizeText,
  metricLabels: METRIC_LABELS,
  palette,
  renderLegendChips,
  safeJson,
  t,
});
const workbenchController = createWorkbenchController({
  appState: state,
  studioDom: dom,
  dom: workbenchDom,
  state: workbenchState,
  clearCanvas,
  flushActiveNumericInput,
  postJson,
  fetchJson,
  buildQuery,
  safeJson,
  escapeHtml,
  setStatus,
  t,
  trainingMonitor,
  renderPreview,
  renderWorkbenchInspectorState: () => workbenchInspector.renderState(),
  renderWorkbenchVisuals: () => workbenchInspector.renderVisuals(),
  clearWorkbenchInspectorState: ({ clearJson = false } = {}) => workbenchInspector.clearState({ clearJson }),
  buildOverwriteConfirmMessage,
});

init().catch((error) => {
  console.error(error);
  setStatus(error.message ?? t("status.loadFailed"), "error");
});

