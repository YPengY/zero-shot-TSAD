/**
 * Thin facade that assembles formatting rules and KPI selection into one
 * training-monitor presenter API.
 */

import { createTrainingFormatting } from "./training_formatting.js";
import { buildTrainingViewModel } from "./training_kpis.js";

/**
 * Build the presenter used by the training monitor renderer.
 */
export function createTrainingPresenter(deps) {
  const {
    formatMetric,
    getLocale,
    localizeText,
    metricLabels,
    t,
  } = deps;

  const formatting = createTrainingFormatting({
    getLocale,
    localizeText,
    metricLabels,
    t,
  });

  function buildViewModel({ job, metrics }) {
    return buildTrainingViewModel({
      job,
      metrics,
      formatting,
      formatMetric,
      localizeText,
      t,
    });
  }

  return {
    buildViewModel,
    formatSplitMetricLabel: formatting.formatSplitMetricLabel,
  };
}
