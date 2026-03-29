/**
 * Training monitor DOM rendering for the workbench UI.
 *
 * The monitor consumes a presenter-generated view model and turns it into DOM
 * updates and charts. It intentionally does not own networking or metric
 * selection rules.
 */

import { createTrainingPresenter } from "./training_presenter.js";

/**
 * Build the training monitor renderer from DOM and chart dependencies.
 */
export function createTrainingMonitor(deps) {
  const {
    dom,
    state,
    clearCanvas,
    drawMetricChart,
    escapeHtml,
    formatMetric,
    getLocale,
    localizeText,
    metricLabels,
    palette,
    renderLegendChips,
    safeJson,
    t,
  } = deps;

  const presenter = createTrainingPresenter({
    formatMetric,
    getLocale,
    localizeText,
    metricLabels,
    t,
  });

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

  function reset() {
    stopPolling("train");
    state.trainingJob = null;
    state.trainingMetrics = null;
    dom.trainingStatus.innerHTML = "";
    dom.trainingStage.textContent = t("workbench.trainingIdle");
    dom.trainingProgressFill.style.width = "0%";
    dom.trainingProgressMeta.innerHTML = "";
    dom.trainingKpis.innerHTML = buildEmptyStateMarkup(t("workbench.trainingIdle"));
    renderLegendChips(dom.lossLegend, []);
    renderLegendChips(dom.qualityLegend, []);
    renderLegendChips(dom.calibrationLegend, []);
    clearCanvas(dom.lossCanvas);
    clearCanvas(dom.qualityCanvas);
    clearCanvas(dom.calibrationCanvas);
    dom.trainingSummary.textContent = t("workbench.trainingIdle");
    dom.trainingLog.textContent = t("workbench.trainingIdle");
  }

  function setJob(job) {
    state.trainingJob = job;
    render();
  }

  /**
   * Render the latest job and metrics snapshot into the monitor surface.
   */
  function render() {
    const viewModel = presenter.buildViewModel({
      job: state.trainingJob,
      metrics: state.trainingMetrics,
    });

    dom.trainingStatus.innerHTML = renderStatPills(viewModel.statusChips);
    dom.trainingStage.textContent = viewModel.stageText;
    dom.trainingProgressFill.style.width = `${(viewModel.progressRatio * 100).toFixed(1)}%`;
    dom.trainingProgressMeta.innerHTML = renderStatPills(viewModel.progressMeta);

    renderTrainingKpis(viewModel.kpiCards);
    renderTrainingCharts(state.trainingMetrics);

    dom.trainingSummary.textContent = viewModel.summaryPayload
      ? safeJson(viewModel.summaryPayload)
      : t("workbench.trainingIdle");
    dom.trainingLog.textContent = viewModel.logText;
  }

  function renderTrainingKpis(cards) {
    if (!cards.length) {
      dom.trainingKpis.innerHTML = buildEmptyStateMarkup(t("workbench.trainingIdle"));
      return;
    }
    dom.trainingKpis.innerHTML = cards
      .map(
        (card) => `
          <section class="training-kpi-card" data-tone="${escapeHtml(card.tone ?? "neutral")}">
            <div class="training-kpi-label">${escapeHtml(card.label)}</div>
            <div class="training-kpi-value">${escapeHtml(card.value)}</div>
            <div class="training-kpi-note">${escapeHtml(card.note)}</div>
          </section>
        `,
      )
      .join("");
  }

  function renderTrainingCharts(metricsPayload) {
    const historyView = metricsPayload?.history_view ?? {};
    renderTrainingChart(
      dom.lossCanvas,
      dom.lossLegend,
      historyView,
      historyView.chart_groups?.loss ?? (historyView.preferred_loss ? [historyView.preferred_loss] : []),
    );
    renderTrainingChart(
      dom.qualityCanvas,
      dom.qualityLegend,
      historyView,
      historyView.chart_groups?.quality ?? (historyView.preferred_quality ? [historyView.preferred_quality] : []),
    );
    renderTrainingChart(
      dom.calibrationCanvas,
      dom.calibrationLegend,
      historyView,
      historyView.chart_groups?.calibration ?? [],
    );
  }

  /**
   * Render a chart group only when the requested metric series exist.
   */
  function renderTrainingChart(canvas, legendContainer, historyView, metricNames) {
    const activeMetricNames = (metricNames ?? []).filter((name) => Array.isArray(historyView?.series?.[name]));
    if (!activeMetricNames.length) {
      clearCanvas(canvas);
      renderLegendChips(legendContainer, []);
      return;
    }

    drawMetricChart(canvas, historyView.epochs ?? [], historyView.series ?? {}, activeMetricNames);
    renderLegendChips(
      legendContainer,
      activeMetricNames.map((name, index) => ({
        color: palette[index % palette.length],
        label: presenter.formatSplitMetricLabel(name),
      })),
    );
  }

  function renderStatPills(values) {
    return values
      .map((value) => `<span class="stat-pill">${escapeHtml(String(value))}</span>`)
      .join("");
  }

  function buildEmptyStateMarkup(message) {
    return `<div class="empty-state">${escapeHtml(message)}</div>`;
  }

  return {
    render,
    reset,
    setJob,
    stopPolling,
  };
}
