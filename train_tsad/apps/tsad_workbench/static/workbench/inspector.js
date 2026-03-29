/**
 * Render workbench dataset inspection state independently from networking
 * and controller orchestration.
 */

/**
 * Build the renderer responsible for packed-sample and packed-window inspection.
 */
export function createWorkbenchInspector(deps) {
  const {
    dom,
    state,
    clearCanvas,
    drawPatchBarChart,
    drawSeriesPairChart,
    escapeHtml,
    formatMetric,
    localizeText,
    t,
  } = deps;

  /**
   * Render inspector summary pills and warnings from the active payload.
   */
  function renderState() {
    if (state.activeInspector === "window" && state.currentWindow) {
      renderSampleStats(buildWindowStats(state.currentWindow));
      setWorkbenchWarning(buildPatchAlignmentWarning(state.currentWindow));
      return;
    }
    if (state.activeInspector === "sample" && state.currentSample) {
      renderSampleStats(buildSampleStats(state.currentSample));
      setWorkbenchWarning("");
      return;
    }
    renderSampleStats([]);
    setWorkbenchWarning("");
  }

  /**
   * Render the current sample/window plots without mutating controller state.
   */
  function renderVisuals() {
    if (state.currentSample) {
      drawSeriesPairChart(
        dom.sampleCanvas,
        state.currentSample.series_feature ?? [],
        state.currentSample.normal_series_feature ?? [],
        state.currentSample.point_mask_feature ?? [],
        [],
      );
    } else {
      clearCanvas(dom.sampleCanvas);
    }

    if (state.currentWindow) {
      drawSeriesPairChart(
        dom.windowCanvas,
        state.currentWindow.series_feature ?? [],
        state.currentWindow.normal_series_feature ?? [],
        state.currentWindow.point_mask_feature ?? [],
        state.currentWindow.padding_mask ?? [],
      );
      drawPatchBarChart(dom.patchCanvas, state.currentWindow.patch_labels ?? []);
    } else {
      clearCanvas(dom.windowCanvas);
      clearCanvas(dom.patchCanvas);
    }
  }

  function clearState(options = {}) {
    const { clearJson = false } = options;
    state.currentSample = null;
    state.currentWindow = null;
    state.activeInspector = null;
    renderState();
    renderVisuals();
    if (clearJson) {
      dom.datasetJson.textContent = t("workbench.samplesEmpty");
    }
  }

  function renderSampleStats(values) {
    dom.sampleStats.innerHTML = (values ?? [])
      .map((value) => `<span class="stat-pill">${escapeHtml(String(value))}</span>`)
      .join("");
  }

  function setWorkbenchWarning(message) {
    const text = String(message ?? "").trim();
    dom.datasetWarning.textContent = text;
    dom.datasetWarning.classList.toggle("hidden", text.length === 0);
  }

  function buildSampleStats(payload) {
    const stats = [
      payload.sample_id,
      `${t("workbench.label.length")} ${payload.length}`,
      `${t("workbench.label.feature")} ${payload.feature_index}`,
      `${t("workbench.label.ratio")} ${formatMetric(payload.anomaly_ratio_feature)}`,
      `${t("workbench.label.windows")} ${payload.windowing?.num_windows ?? 0}`,
    ];
    if ((payload.windowing?.padded_window_count ?? 0) > 0) {
      stats.push(`${t("workbench.label.padded")} ${payload.windowing.padded_window_count}`);
    }
    if ((payload.windowing?.tail_window_count ?? 0) > 0) {
      stats.push(`${t("workbench.label.tail")} ${payload.windowing.tail_window_count}`);
    }
    if ((payload.windowing?.short_window_count ?? 0) > 0) {
      stats.push(`${t("workbench.label.short")} ${payload.windowing.short_window_count}`);
    }
    return stats;
  }

  function buildWindowStats(payload) {
    const totalPatches = payload.total_patch_count ?? payload.patch_labels?.length ?? 0;
    const stats = [
      payload.sample_id,
      `${t("workbench.label.window")} ${payload.window_index}`,
      `[${payload.start}, ${payload.end})`,
      `${t("workbench.label.effective")} ${payload.effective_length}`,
      `${t("workbench.label.patches")} ${totalPatches}`,
    ];
    if ((payload.padded_steps ?? 0) > 0) {
      stats.push(`${t("workbench.label.padded")} ${payload.padded_steps}`);
    }
    if (
      typeof payload.valid_patch_count === "number" &&
      totalPatches > 0 &&
      payload.valid_patch_count < totalPatches
    ) {
      stats.push(
        `${t("workbench.label.validPatches")} ${payload.valid_patch_count}/${totalPatches}`,
      );
    }
    if (payload.is_short_window) {
      stats.push(t("workbench.label.short"));
    } else if (payload.is_tail_window) {
      stats.push(t("workbench.label.tail"));
    }
    return stats;
  }

  /**
   * Build a user-facing explanation when padding changes patch semantics.
   */
  function buildPatchAlignmentWarning(payload) {
    const messages = [];
    if (typeof payload?.padded_steps === "number" && payload.padded_steps > 0) {
      if (payload.is_short_window) {
        messages.push(
          localizeText(
            `Short sample is kept as one right-padded window (${payload.padded_steps} padded steps).`,
            `短样本会保留为一个右侧补齐窗口（补齐 ${payload.padded_steps} 个时间步）。`,
          ),
        );
      } else if (payload.is_tail_window) {
        messages.push(
          localizeText(
            `Tail remainder is kept as one right-padded window (${payload.padded_steps} padded steps).`,
            `尾部剩余片段会保留为一个右侧补齐窗口（补齐 ${payload.padded_steps} 个时间步）。`,
          ),
        );
      } else {
        messages.push(
          localizeText(
            `This window is right-padded by ${payload.padded_steps} steps.`,
            `当前窗口右侧补齐了 ${payload.padded_steps} 个时间步。`,
          ),
        );
      }
      if (
        typeof payload.valid_patch_count === "number" &&
        typeof payload.total_patch_count === "number" &&
        payload.total_patch_count > 0
      ) {
        messages.push(
          localizeText(
            `Only the first ${payload.valid_patch_count}/${payload.total_patch_count} patches are fully observed before padding.`,
            `补齐前只有前 ${payload.valid_patch_count}/${payload.total_patch_count} 个 Patch 完全由真实观测组成。`,
          ),
        );
      }
    }
    if (payload?.patch_alignment_warning) {
      messages.push(
        localizeText(
          `Patch labels only cover complete patches because context_size (${payload.context_size}) is not divisible by patch_size (${payload.patch_size}).`,
          `由于 context_size (${payload.context_size}) 不能被 patch_size (${payload.patch_size}) 整除，Patch 标签只覆盖完整 patch，末尾残余步长不会计入标签。`,
        ),
      );
    }
    return messages.join(" ");
  }

  return {
    clearState,
    renderState,
    renderVisuals,
  };
}
