/**
 * Workbench orchestration for preview galleries, dataset browsing, generation jobs,
 * and training job polling.
 *
 * This module is the shell for workbench behavior: it coordinates DOM events,
 * HTTP requests, and renderer updates, while leaving chart rendering and
 * inspector/training presentation to dedicated submodules.
 */

/**
 * Build the workbench controller from injected UI and networking dependencies.
 */
export function createWorkbenchController(deps) {
  const {
    appState,
    studioDom,
    dom,
    state,
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
    renderWorkbenchInspectorState,
    renderWorkbenchVisuals,
    clearWorkbenchInspectorState,
    buildOverwriteConfirmMessage,
  } = deps;

  let eventsBound = false;

  /**
   * Apply bootstrap defaults and wire one-time event listeners.
   */
  function initialize(bootstrap) {
    state.bootstrap = bootstrap ?? {};
    const defaults = state.bootstrap.generation_defaults ?? {};
    dom.runName.value = state.bootstrap.default_run_name ?? "workbench_run";
    dom.trainSamples.value = String(defaults.train_samples ?? 10000);
    dom.valSamples.value = String(defaults.val_samples ?? 1500);
    dom.testSamples.value = String(defaults.test_samples ?? 1500);
    dom.samplesPerShard.value = String(defaults.samples_per_shard ?? 128);
    dom.seedBase.value = String(defaults.seed_base ?? Number(appState.config?.seed ?? 0));
    dom.trainDevice.value = String(defaults.train_device ?? "cuda");
    dom.trainEpochs.value = String(defaults.train_max_epochs ?? 5);
    dom.trainDeviceOverride.value = String(defaults.train_device ?? "cuda");
    dom.trainEpochOverride.value = String(defaults.train_max_epochs ?? 5);

    const templates = state.bootstrap.train_templates ?? [];
    dom.trainTemplate.innerHTML = templates
      .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
      .join("");
    dom.trainTemplate.value = state.bootstrap.default_train_template ?? templates[0] ?? "";
    dom.overwriteRun.checked = false;
    dom.previewGallery.innerHTML = buildEmptyStateMarkup(t("workbench.galleryEmpty"));
    dom.datasetJson.textContent = t("workbench.samplesEmpty");

    clearCanvas(dom.sampleCanvas);
    clearCanvas(dom.windowCanvas);
    clearCanvas(dom.patchCanvas);
    clearCanvas(dom.lossCanvas);
    clearCanvas(dom.qualityCanvas);
    clearCanvas(dom.calibrationCanvas);
    trainingMonitor.reset();
    syncWindowStep();
    bindEvents();
  }

  function refreshLocale() {
    renderPreviewGallery();
    trainingMonitor.render();
    if (!state.sampleList.length && state.activeInspector == null) {
      dom.datasetJson.textContent = t("workbench.samplesEmpty");
    }
  }

  function bindEvents() {
    if (eventsBound) {
      return;
    }
    eventsBound = true;

    dom.previewBatch?.addEventListener("click", buildPreviewGallery);
    dom.previewGenerate?.addEventListener("click", buildPreviewGallery);
    dom.generateSubmit?.addEventListener("click", startGenerationJob);
    dom.loadRun?.addEventListener("click", () => loadRunInfo(true));
    dom.loadSamples?.addEventListener("click", loadSamplesForSplit);
    dom.previewPackedSample?.addEventListener("click", previewPackedSample);
    dom.previewWindow?.addEventListener("click", previewPackedWindow);
    dom.contextSize?.addEventListener("input", syncWindowStep);
    dom.contextSize?.addEventListener("change", syncWindowStep);
    dom.startTraining?.addEventListener("click", startTrainingJob);
    dom.refreshTraining?.addEventListener("click", () => refreshTrainingMetrics());
    dom.previewGallery?.addEventListener("click", async (event) => {
      const target = event.target instanceof HTMLElement ? event.target.closest("[data-preview-id]") : null;
      if (target) {
        await loadPreviewById(String(target.getAttribute("data-preview-id") || ""));
      }
    });
  }

  function syncWindowStep() {
    if (!dom.contextSize || !dom.stride) {
      return;
    }
    const contextSize = Math.max(1, Number(dom.contextSize.value || 1));
    dom.stride.value = String(contextSize);
  }

  async function buildPreviewGallery() {
    if (!flushActiveNumericInput()) {
      return;
    }
    try {
      setStatus(t("status.previewing"), "ok");
      const payload = await postJson("/api/preview-batch", {
        config: appState.config,
        count: Number(dom.previewCount.value || 6),
        seed_base: Number(dom.previewSeed.value || appState.config.seed || 0),
      });
      state.previewCards = payload.previews ?? [];
      renderPreviewGallery();
      setStatus(t("workbench.galleryLoaded"), "ok");
    } catch (error) {
      setStatus(error.message ?? t("status.previewing"), "error");
    }
  }

  function renderPreviewGallery() {
    if (!state.previewCards.length) {
      dom.previewGallery.innerHTML = buildEmptyStateMarkup(t("workbench.galleryEmpty"));
      return;
    }
    dom.previewGallery.innerHTML = state.previewCards
      .map((card) => {
        const isActive = card.preview_id === state.activePreviewId;
        return `
          <button type="button" class="preview-gallery-card ${isActive ? "active" : ""}" data-preview-id="${escapeHtml(card.preview_id)}">
            <h4>${escapeHtml(t("workbench.label.seed"))} ${escapeHtml(String(card.seed))}</h4>
            <div class="chip-row">
              <span class="preview-chip">${escapeHtml(t("workbench.label.lengthShort"))}=${escapeHtml(String(card.length))}</span>
              <span class="preview-chip">${escapeHtml(t("workbench.label.seriesShort"))}=${escapeHtml(String(card.num_series))}</span>
            </div>
            <div class="chip-row">
              <span class="preview-chip">${escapeHtml(t("workbench.label.events"))} ${escapeHtml(String(card.num_events))}</span>
              <span class="preview-chip">${escapeHtml(t("workbench.label.local"))} ${escapeHtml(String(card.num_local_events))}</span>
              <span class="preview-chip">${escapeHtml(t("workbench.label.seasonal"))} ${escapeHtml(String(card.num_seasonal_events))}</span>
            </div>
          </button>
        `;
      })
      .join("");
  }

  async function loadPreviewById(previewId) {
    if (!previewId) {
      return;
    }
    try {
      const payload = await fetchJson(`/api/preview-item?preview_id=${encodeURIComponent(previewId)}`);
      state.activePreviewId = previewId;
      appState.preview = payload.preview;
      appState.hoveredEventIndex = null;
      appState.pinnedEventIndex = null;
      appState.selectedNodes = Array.from(
        {
          length: Math.min(
            Number(studioDom.maxNodesSelect.value || 6),
            Number(appState.preview?.summary?.num_series || 0),
          ),
        },
        (_, index) => index,
      );
      renderPreview();
      renderPreviewGallery();
      setStatus(t("status.previewed"), "ok");
    } catch (error) {
      setStatus(error.message ?? t("status.previewing"), "error");
    }
  }

  /**
   * Collect the dataset-generation payload from the current form state.
   *
   * This keeps form-to-request translation in one place so controller actions
   * and tests can reason about generation settings consistently.
   */
  function collectGenerationPayload() {
    const defaults = state.bootstrap.generation_defaults ?? {};
    return {
      config: appState.config,
      run_name: dom.runName.value.trim(),
      train_samples: Number(dom.trainSamples.value || 10000),
      val_samples: Number(dom.valSamples.value || 1500),
      test_samples: Number(dom.testSamples.value || 1500),
      direct_pack: defaults.direct_pack ?? true,
      window_pack: defaults.window_pack ?? false,
      direct_window_pack: defaults.direct_window_pack ?? true,
      window_context_size: Number(defaults.window_context_size ?? 1024),
      window_patch_size: Number(defaults.window_patch_size ?? 16),
      window_stride: defaults.window_stride ?? 1024,
      window_windows_per_shard: Number(defaults.window_windows_per_shard ?? 4096),
      window_include_tail: defaults.window_include_tail ?? true,
      window_pad_short_sequences: defaults.window_pad_short_sequences ?? true,
      window_debug_sidecar: defaults.window_debug_sidecar ?? false,
      window_train_min_patch_positive_ratio: defaults.window_train_min_patch_positive_ratio ?? null,
      window_train_min_anomaly_point_ratio: defaults.window_train_min_anomaly_point_ratio ?? null,
      samples_per_shard: Number(dom.samplesPerShard.value || 128),
      seed_base: Number(dom.seedBase.value || appState.config.seed || 0),
      train_template: dom.trainTemplate.value,
      train_device: dom.trainDevice.value.trim(),
      train_max_epochs: Number(dom.trainEpochs.value || 5),
      train_batch_size: Number(dom.trainBatchSize.value || 16),
      eval_batch_size: Number(dom.evalBatchSize.value || 16),
      overwrite_run: dom.overwriteRun.checked,
    };
  }

  async function startGenerationJob() {
    if (!flushActiveNumericInput()) {
      return;
    }
    try {
      const payload = collectGenerationPayload();
      if (payload.overwrite_run) {
        const confirmed = window.confirm(buildOverwriteConfirmMessage(payload.run_name));
        if (!confirmed) {
          return;
        }
        payload.confirm_overwrite = true;
      }
      const response = await postJson("/api/generate", payload);
      setStatus(t("workbench.generationStarted"), "ok");
      pollJob("generate", response.job_id);
    } catch (error) {
      setStatus(error.message ?? t("workbench.generationStarted"), "error");
    }
  }

  async function loadRunInfo(showStatus = false) {
    const runPath = dom.runPath.value.trim();
    if (!runPath) {
      if (showStatus) {
        setStatus(t("workbench.error.missingRunLookup"), "error");
      }
      return;
    }
    try {
      const info = await fetchJson(`/api/run?path=${encodeURIComponent(runPath)}`);
      applyRunInfo(info);
      if (showStatus) {
        setStatus(t("workbench.runLoaded"), "ok");
      }
    } catch (error) {
      if (showStatus) {
        setStatus(error.message ?? t("workbench.runLoaded"), "error");
      }
    }
  }

  /**
   * Apply run metadata and fan it out to the dependent inspector panes.
   */
  function applyRunInfo(info) {
    state.runInfo = info;
    dom.runPath.value = info.run_root ?? info.packed_root ?? "";
    dom.trainConfigPath.value = info.train_config_path ?? "";
    dom.trainOutputDir.value = info.train_output_dir ?? "";
    dom.runSummary.textContent = safeJson(info);

    void refreshTrainingMetrics({ silent: true });
    if (info.available_splits?.includes(dom.splitSelect.value)) {
      void loadSamplesForSplit();
    } else if (info.available_splits?.[0]) {
      dom.splitSelect.value = info.available_splits[0];
      void loadSamplesForSplit();
    }
  }

  async function loadSamplesForSplit() {
    const runPath = dom.runPath.value.trim();
    if (!runPath) {
      return;
    }
    state.sampleList = [];
    dom.sampleSelect.innerHTML = "";
    clearWorkbenchInspectorState({ clearJson: true });
    try {
      const payload = await fetchJson(
        `/api/samples?run_root=${encodeURIComponent(runPath)}&split=${encodeURIComponent(dom.splitSelect.value)}&limit=300`,
      );
      state.sampleList = payload.samples ?? [];
      dom.sampleSelect.innerHTML = state.sampleList
        .map((sample) => `<option value="${escapeHtml(sample.sample_id)}">${escapeHtml(sample.sample_id)}</option>`)
        .join("");
      dom.datasetJson.textContent = safeJson(payload);
      if (state.sampleList[0]) {
        dom.sampleSelect.value = state.sampleList[0].sample_id;
      }
    } catch (error) {
      setStatus(error.message ?? t("workbench.samplesEmpty"), "error");
    }
  }

  async function previewPackedSample() {
    const sampleId = dom.sampleSelect.value;
    if (!sampleId) {
      return;
    }
    try {
      syncWindowStep();
      const payload = await fetchJson(
        buildQuery("/api/sample", {
          run_root: dom.runPath.value.trim(),
          split: dom.splitSelect.value,
          sample_id: sampleId,
          feature_index: Number(dom.featureIndex.value || 0),
          slice_start: Number(dom.sliceStart.value || 0),
          slice_end: Number(dom.sliceEnd.value || 0),
          context_size: Number(dom.contextSize.value || 512),
          stride: Number(dom.stride.value || dom.contextSize.value || 512),
          patch_size: Number(dom.patchSize.value || 16),
        }),
      );
      dom.sampleSelect.value = String(payload.sample_id ?? dom.sampleSelect.value);
      dom.featureIndex.value = String(payload.feature_index ?? dom.featureIndex.value);
      dom.sliceStart.value = String(payload.slice_start ?? dom.sliceStart.value);
      dom.sliceEnd.value = String(payload.slice_end ?? dom.sliceEnd.value);
      dom.contextSize.value = String(payload.windowing?.context_size ?? dom.contextSize.value);
      dom.stride.value = String(payload.windowing?.stride ?? dom.stride.value);
      dom.patchSize.value = String(payload.windowing?.patch_size ?? dom.patchSize.value);
      state.currentSample = payload;
      state.currentWindow = null;
      state.activeInspector = "sample";
      renderWorkbenchInspectorState();
      renderWorkbenchVisuals();
      dom.datasetJson.textContent = safeJson(payload);
      setStatus(t("workbench.sampleLoaded"), "ok");
    } catch (error) {
      setStatus(error.message ?? t("workbench.sampleLoaded"), "error");
    }
  }

  async function previewPackedWindow() {
    const sampleId = dom.sampleSelect.value;
    if (!sampleId) {
      return;
    }
    try {
      syncWindowStep();
      const payload = await fetchJson(
        buildQuery("/api/window", {
          run_root: dom.runPath.value.trim(),
          split: dom.splitSelect.value,
          sample_id: sampleId,
          feature_index: Number(dom.featureIndex.value || 0),
          window_index: Number(dom.windowIndex.value || 0),
          context_size: Number(dom.contextSize.value || 512),
          stride: Number(dom.stride.value || dom.contextSize.value || 512),
          patch_size: Number(dom.patchSize.value || 16),
        }),
      );
      dom.sampleSelect.value = String(payload.sample_id ?? dom.sampleSelect.value);
      dom.featureIndex.value = String(payload.feature_index ?? dom.featureIndex.value);
      dom.windowIndex.value = String(payload.window_index ?? dom.windowIndex.value);
      dom.contextSize.value = String(payload.context_size ?? dom.contextSize.value);
      dom.stride.value = String(payload.stride ?? dom.stride.value);
      dom.patchSize.value = String(payload.patch_size ?? dom.patchSize.value);
      if (
        state.currentSample &&
        (
          String(state.currentSample.sample_id) !== String(payload.sample_id) ||
          Number(state.currentSample.feature_index) !== Number(payload.feature_index)
        )
      ) {
        state.currentSample = null;
        clearCanvas(dom.sampleCanvas);
      }
      state.currentWindow = payload;
      state.activeInspector = "window";
      renderWorkbenchInspectorState();
      renderWorkbenchVisuals();
      dom.datasetJson.textContent = safeJson(payload);
      setStatus(t("workbench.windowLoaded"), "ok");
    } catch (error) {
      setStatus(error.message ?? t("workbench.windowLoaded"), "error");
    }
  }

  async function startTrainingJob() {
    const configPath = dom.trainConfigPath.value.trim();
    if (!configPath) {
      setStatus(t("workbench.error.missingTrainConfig"), "error");
      return;
    }
    try {
      const payload = await postJson("/api/train", {
        config_path: configPath,
        device: dom.trainDeviceOverride.value.trim(),
        max_epochs: Number(dom.trainEpochOverride.value || 0) || undefined,
        inspect_data: dom.inspectData.checked,
        inspect_max_samples: Number(dom.inspectMaxSamples.value || 0) || undefined,
      });
      trainingMonitor.reset();
      dom.trainingStage.textContent = t("workbench.trainingStarted");
      setStatus(t("workbench.trainingStarted"), "ok");
      pollJob("train", payload.job_id);
    } catch (error) {
      setStatus(error.message ?? t("workbench.trainingStarted"), "error");
    }
  }

  function stopPolling(kind) {
    const pollState = state.polling[kind];
    if (!pollState) {
      return;
    }
    pollState.disposed = true;
    if (pollState.timerId) {
      window.clearTimeout(pollState.timerId);
    }
    state.polling[kind] = null;
  }

  /**
   * Start polling a background job and route updates to the right UI surface.
   */
  function pollJob(kind, jobId) {
    stopPolling(kind);
    const pollState = {
      disposed: false,
      inFlight: false,
      timerId: null,
      jobId,
    };
    state.polling[kind] = pollState;

    const tick = async () => {
      if (pollState.disposed || pollState.inFlight) {
        return;
      }
      pollState.inFlight = true;
      let shouldContinue = true;
      try {
        const payload = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobId)}`);
        if (pollState.disposed || state.polling[kind] !== pollState) {
          return;
        }

        if (kind === "generate") {
          dom.runSummary.textContent = safeJson(
            payload.result ?? { status: payload.status, logs: payload.logs?.slice(-30) ?? [] },
          );
        } else if (kind === "train") {
          if (payload.artifacts?.output_dir) {
            dom.trainOutputDir.value = payload.artifacts.output_dir;
          }
          trainingMonitor.setJob(payload);
          if (payload.artifacts?.output_dir || dom.trainOutputDir.value.trim() || dom.runPath.value.trim()) {
            await refreshTrainingMetrics({ silent: true });
          }
        }

        if (payload.status === "completed") {
          shouldContinue = false;
          if (kind === "generate" && payload.result) {
            applyRunInfo(payload.result);
          }
          if (kind === "train" && payload.result) {
            dom.trainOutputDir.value = payload.result.output_dir ?? dom.trainOutputDir.value;
            await refreshTrainingMetrics();
          }
        }
        if (payload.status === "failed") {
          shouldContinue = false;
        }
      } catch (error) {
        if (kind === "train") {
          dom.trainingLog.textContent = String(error.message ?? error);
        }
      } finally {
        pollState.inFlight = false;
        if (pollState.disposed || state.polling[kind] !== pollState) {
          return;
        }
        if (shouldContinue) {
          pollState.timerId = window.setTimeout(tick, 2000);
        } else {
          stopPolling(kind);
        }
      }
    };

    void tick();
  }

  /**
   * Refresh persisted training metrics independently from live job polling.
   *
   * This allows the UI to recover state from saved artifacts after reloads or
   * after a completed background process.
   */
  async function refreshTrainingMetrics(options = {}) {
    const { silent = false } = options;
    const outputDir = dom.trainOutputDir.value.trim();
    const runRoot = dom.runPath.value.trim();
    if (!outputDir && !runRoot) {
      return;
    }
    try {
      const payload = await fetchJson(
        buildQuery("/api/train-metrics", {
          output_dir: outputDir,
          run_root: runRoot,
        }),
      );
      state.trainingMetrics = payload;
      if (payload.output_dir) {
        dom.trainOutputDir.value = payload.output_dir;
      }
      trainingMonitor.render();
    } catch (error) {
      if (!silent) {
        dom.trainingSummary.textContent = safeJson({ error: error.message ?? String(error) });
      }
    }
  }

  function buildEmptyStateMarkup(message) {
    return `<div class="empty-state">${escapeHtml(message)}</div>`;
  }

  return {
    initialize,
    refreshLocale,
  };
}
