/**
 * KPI selection and training-monitor view-model assembly.
 */

/**
 * Assemble the view model consumed by the training monitor renderer.
 */
export function buildTrainingViewModel({
  job,
  metrics,
  formatting,
  formatMetric,
  localizeText,
  t,
}) {
  const progress = job?.progress ?? metrics?.progress ?? null;
  const kpis = metrics?.kpis ?? {};

  return {
    statusChips: buildTrainingStatusChips(job, {
      formatTrainingStatus: formatting.formatTrainingStatus,
      t,
    }),
    stageText: formatting.buildTrainingStageText(job, progress, kpis),
    progressRatio: clampUnit(
      progress?.overall_progress_ratio ??
        kpis?.overall_progress_ratio ??
        (job?.status === "completed" ? 1 : 0),
    ),
    progressMeta: buildTrainingProgressMeta(progress, kpis, {
      formatDuration: formatting.formatDuration,
      formatLearningRate: formatting.formatLearningRate,
      formatPercent: formatting.formatPercent,
      formatSplitLabel: formatting.formatSplitLabel,
      localizeText,
      t,
    }),
    kpiCards: buildTrainingKpiCards(job, progress, kpis, {
      buildTrainingStageText: formatting.buildTrainingStageText,
      formatDuration: formatting.formatDuration,
      formatLearningRate: formatting.formatLearningRate,
      formatMetric,
      formatMetricLabel: formatting.formatMetricLabel,
      formatPercent: formatting.formatPercent,
      formatSignedPercent: formatting.formatSignedPercent,
      localizeText,
      t,
    }),
    summaryPayload: buildTrainingSummaryPayload(job, metrics),
    logText: buildTrainingLogText(job, t("workbench.trainingIdle")),
  };
}

function buildTrainingStatusChips(job, helpers) {
  if (!job) {
    return [];
  }
  const { formatTrainingStatus, t } = helpers;
  return [
    `${t("workbench.label.job")} ${job.job_id}`,
    `${t("workbench.label.status")} ${formatTrainingStatus(job.status)}`,
    job.started_at
      ? `${t("workbench.label.started")} ${new Date(job.started_at * 1000).toLocaleTimeString()}`
      : null,
    job.finished_at
      ? `${t("workbench.label.finished")} ${new Date(job.finished_at * 1000).toLocaleTimeString()}`
      : null,
  ].filter(Boolean);
}

function buildTrainingProgressMeta(progress, kpis, helpers) {
  const {
    formatDuration,
    formatLearningRate,
    formatPercent,
    formatSplitLabel,
    localizeText,
    t,
  } = helpers;

  const values = [];
  const epochCurrent = progress?.epoch_current ?? kpis?.epoch_current;
  const epochTotal = progress?.epoch_total ?? kpis?.epoch_total;

  if (epochCurrent != null || epochTotal != null) {
    values.push(`${t("workbench.kpi.epoch")} ${epochCurrent ?? 0}/${epochTotal ?? "-"}`);
  }
  if (progress?.split) {
    values.push(
      `${formatSplitLabel(progress.split)} ${progress.split_step_current ?? 0}/${progress.split_step_total ?? "-"}`,
    );
  }
  if (typeof progress?.learning_rate === "number") {
    values.push(`${t("workbench.kpi.learningRate")} ${formatLearningRate(progress.learning_rate)}`);
  }
  if (typeof progress?.elapsed_seconds === "number") {
    values.push(`${localizeText("Elapsed", "\u5df2\u8017\u65f6")} ${formatDuration(progress.elapsed_seconds)}`);
  }
  if (typeof progress?.eta_seconds === "number" && Number.isFinite(progress.eta_seconds)) {
    values.push(`${localizeText("ETA", "\u9884\u8ba1\u5269\u4f59")} ${formatDuration(progress.eta_seconds)}`);
  }
  if (typeof progress?.overall_progress_ratio === "number") {
    values.push(formatPercent(progress.overall_progress_ratio));
  }
  return values;
}

/**
 * Build the KPI cards that summarize the current run state.
 */
function buildTrainingKpiCards(job, progress, kpis, helpers) {
  const {
    buildTrainingStageText,
    formatDuration,
    formatLearningRate,
    formatMetric,
    formatMetricLabel,
    formatPercent,
    formatSignedPercent,
    localizeText,
    t,
  } = helpers;

  const currentSplitMetrics = progress?.current_split_metrics ?? {};
  const latestTrain = kpis?.latest_train ?? progress?.latest_train_metrics ?? {};
  const latestVal = kpis?.latest_val ?? progress?.latest_val_metrics ?? {};
  const monitorMetric = kpis?.monitor_metric ?? progress?.monitor_metric ?? null;
  const bestMetric = kpis?.best_metric ?? progress?.best_metric ?? null;
  const latestMonitorValue = kpis?.latest_monitor_value ?? null;
  const epochCurrent = progress?.epoch_current ?? kpis?.epoch_current ?? null;
  const epochTotal = progress?.epoch_total ?? kpis?.epoch_total ?? null;
  const trainLoss = firstDefinedMetric(
    progress?.split === "train" ? currentSplitMetrics.total_loss : null,
    latestTrain.total_loss,
  );
  const qualityMetricName = pickPreferredMetricName(latestVal, [
    "f1",
    "pr_auc",
    "precision",
    "recall",
    "patch_accuracy",
    "point_accuracy",
    "total_loss",
  ]);
  const qualityMetricValue = qualityMetricName ? latestVal?.[qualityMetricName] : null;
  const predictedPositiveRate = firstDefinedMetric(
    progress?.split === "train" ? currentSplitMetrics.predicted_positive_rate : null,
    latestTrain.predicted_positive_rate,
    latestVal.predicted_positive_rate,
  );
  const targetPositiveRate = firstDefinedMetric(
    progress?.split === "train" ? currentSplitMetrics.target_positive_rate : null,
    latestTrain.target_positive_rate,
    latestVal.target_positive_rate,
  );
  const thresholdValue = firstDefinedMetric(currentSplitMetrics.threshold, latestVal.threshold);
  const pointAccuracy = firstDefinedMetric(
    currentSplitMetrics.point_accuracy,
    latestVal.point_accuracy,
    latestTrain.point_accuracy,
  );
  const pointLoss = firstDefinedMetric(
    currentSplitMetrics.point_anomaly_loss,
    latestTrain.point_anomaly_loss,
    latestVal.point_anomaly_loss,
  );
  const pointPredictedPositiveRate = firstDefinedMetric(
    currentSplitMetrics.point_predicted_positive_rate,
    latestTrain.point_predicted_positive_rate,
    latestVal.point_predicted_positive_rate,
  );
  const pointTargetPositiveRate = firstDefinedMetric(
    currentSplitMetrics.point_target_positive_rate,
    latestTrain.point_target_positive_rate,
    latestVal.point_target_positive_rate,
  );

  return [
    {
      label: t("workbench.kpi.epoch"),
      value: `${epochCurrent ?? 0} / ${epochTotal ?? "-"}`,
      note: buildTrainingStageText(job, progress, kpis),
      tone: "accent",
    },
    {
      label: monitorMetric
        ? `${t("workbench.kpi.monitor")} (${formatMetricLabel(monitorMetric)})`
        : t("workbench.kpi.monitor"),
      value: formatMetric(bestMetric),
      note:
        latestMonitorValue != null
          ? `${localizeText("Latest", "\u6700\u65b0")} ${formatMetric(latestMonitorValue)}`
          : localizeText(
              "Waiting for the first monitored value.",
              "\u7b49\u5f85\u9996\u4e2a\u76d1\u63a7\u503c\u3002",
            ),
      tone: "blue",
    },
    {
      label: t("workbench.kpi.trainLoss"),
      value: formatMetric(trainLoss),
      note: buildTrainLossNote(latestTrain, {
        formatMetric,
        formatMetricLabel,
        formatPercent,
        localizeText,
      }),
      tone: "accent",
    },
    qualityMetricName
      ? {
          label: `${t("workbench.kpi.valQuality")} (${formatMetricLabel(qualityMetricName)})`,
          value: qualityMetricName.includes("accuracy")
            ? formatPercent(qualityMetricValue)
            : formatMetric(qualityMetricValue),
          note: buildValidationQualityNote(latestVal, {
            formatMetric,
            formatMetricLabel,
            formatPercent,
            localizeText,
          }),
          tone: "blue",
        }
      : null,
    pointAccuracy != null || pointLoss != null
      ? {
          label: t("workbench.kpi.pointHead"),
          value: pointAccuracy != null ? formatPercent(pointAccuracy) : formatMetric(pointLoss),
          note: buildPointAuxNote(
            {
              point_anomaly_loss: pointLoss,
              point_accuracy: pointAccuracy,
              point_predicted_positive_rate: pointPredictedPositiveRate,
              point_target_positive_rate: pointTargetPositiveRate,
            },
            {
              formatMetric,
              formatMetricLabel,
              formatPercent,
              localizeText,
            },
          ),
          tone: "blue",
        }
      : null,
    {
      label: t("workbench.kpi.learningRate"),
      value: formatLearningRate(progress?.learning_rate ?? kpis?.learning_rate),
      note: localizeText(
        "Updated from the current optimizer state.",
        "\u53d6\u81ea\u5f53\u524d\u4f18\u5316\u5668\u72b6\u6001\u3002",
      ),
      tone: "neutral",
    },
    predictedPositiveRate != null && targetPositiveRate != null
      ? {
          label: t("workbench.kpi.balance"),
          value: formatSignedPercent(predictedPositiveRate - targetPositiveRate),
          note: `${localizeText("pred", "\u9884\u6d4b")} ${formatPercent(predictedPositiveRate)} | ${localizeText("target", "\u76ee\u6807")} ${formatPercent(targetPositiveRate)}`,
          tone: "neutral",
        }
      : null,
    thresholdValue != null
      ? {
          label: t("workbench.kpi.threshold"),
          value: formatMetric(thresholdValue),
          note: localizeText(
            "Latest validation threshold snapshot.",
            "\u6700\u65b0\u9a8c\u8bc1\u9608\u503c\u5feb\u7167\u3002",
          ),
          tone: "neutral",
        }
      : null,
    {
      label: t("workbench.kpi.runtime"),
      value: formatDuration(progress?.elapsed_seconds ?? kpis?.elapsed_seconds),
      note:
        typeof (progress?.eta_seconds ?? kpis?.eta_seconds) === "number"
          ? `${localizeText("ETA", "\u9884\u8ba1\u5269\u4f59")} ${formatDuration(progress?.eta_seconds ?? kpis?.eta_seconds)}`
          : localizeText("No ETA yet.", "\u6682\u65e0\u5269\u4f59\u65f6\u95f4\u4f30\u8ba1\u3002"),
      tone: "neutral",
    },
  ].filter(Boolean);
}

function buildTrainLossNote(metrics, helpers) {
  const { formatMetric, formatMetricLabel, formatPercent, localizeText } = helpers;
  if (!metrics || typeof metrics !== "object") {
    return localizeText(
      "Waiting for the first train epoch.",
      "\u7b49\u5f85\u9996\u4e2a\u8bad\u7ec3\u8f6e\u6b21\u7ed3\u679c\u3002",
    );
  }
  const parts = [];
  if (typeof metrics.anomaly_loss === "number") {
    parts.push(`${formatMetricLabel("anomaly_loss")} ${formatMetric(metrics.anomaly_loss)}`);
  }
  if (typeof metrics.reconstruction_loss === "number") {
    parts.push(`${formatMetricLabel("reconstruction_loss")} ${formatMetric(metrics.reconstruction_loss)}`);
  }
  if (typeof metrics.patch_accuracy === "number") {
    parts.push(`${formatMetricLabel("patch_accuracy")} ${formatPercent(metrics.patch_accuracy)}`);
  }
  if (typeof metrics.point_anomaly_loss === "number") {
    parts.push(`${formatMetricLabel("point_anomaly_loss")} ${formatMetric(metrics.point_anomaly_loss)}`);
  }
  if (typeof metrics.point_accuracy === "number") {
    parts.push(`${formatMetricLabel("point_accuracy")} ${formatPercent(metrics.point_accuracy)}`);
  }
  return parts.join(" | ") || localizeText(
    "Train metrics are accumulating.",
    "\u8bad\u7ec3\u6307\u6807\u6b63\u5728\u7d2f\u79ef\u3002",
  );
}

function buildValidationQualityNote(metrics, helpers) {
  const { formatMetric, formatMetricLabel, formatPercent, localizeText } = helpers;
  if (!metrics || typeof metrics !== "object") {
    return localizeText(
      "Validation has not finished yet.",
      "\u9a8c\u8bc1\u8fd8\u6ca1\u6709\u7ed3\u675f\u3002",
    );
  }
  const parts = [];
  if (typeof metrics.patch_accuracy === "number") {
    parts.push(`${formatMetricLabel("patch_accuracy")} ${formatPercent(metrics.patch_accuracy)}`);
  }
  if (typeof metrics.precision === "number") {
    parts.push(`${formatMetricLabel("precision")} ${formatMetric(metrics.precision)}`);
  }
  if (typeof metrics.recall === "number") {
    parts.push(`${formatMetricLabel("recall")} ${formatMetric(metrics.recall)}`);
  }
  if (typeof metrics.pr_auc === "number") {
    parts.push(`${formatMetricLabel("pr_auc")} ${formatMetric(metrics.pr_auc)}`);
  }
  if (typeof metrics.point_accuracy === "number") {
    parts.push(`${formatMetricLabel("point_accuracy")} ${formatPercent(metrics.point_accuracy)}`);
  }
  return parts.join(" | ") || localizeText(
    "Only the latest validation snapshot is available.",
    "\u5f53\u524d\u53ea\u62ff\u5230\u4e86\u6700\u65b0\u4e00\u4efd\u9a8c\u8bc1\u5feb\u7167\u3002",
  );
}

function buildPointAuxNote(metrics, helpers) {
  const { formatMetric, formatMetricLabel, formatPercent, localizeText } = helpers;
  if (!metrics || typeof metrics !== "object") {
    return localizeText(
      "Observation-space metrics are accumulating.",
      "\u89c2\u6d4b\u7a7a\u95f4\u8f85\u52a9\u6307\u6807\u6b63\u5728\u7d2f\u79ef\u3002",
    );
  }
  const parts = [];
  if (typeof metrics.point_anomaly_loss === "number") {
    parts.push(`${formatMetricLabel("point_anomaly_loss")} ${formatMetric(metrics.point_anomaly_loss)}`);
  }
  if (typeof metrics.point_accuracy === "number") {
    parts.push(`${formatMetricLabel("point_accuracy")} ${formatPercent(metrics.point_accuracy)}`);
  }
  if (
    typeof metrics.point_predicted_positive_rate === "number" &&
    typeof metrics.point_target_positive_rate === "number"
  ) {
    parts.push(
      `${localizeText("pred", "\u9884\u6d4b")} ${formatPercent(metrics.point_predicted_positive_rate)} | ${localizeText("target", "\u76ee\u6807")} ${formatPercent(metrics.point_target_positive_rate)}`,
    );
  }
  return parts.join(" | ") || localizeText(
    "Observation-space metrics are accumulating.",
    "\u89c2\u6d4b\u7a7a\u95f4\u8f85\u52a9\u6307\u6807\u6b63\u5728\u7d2f\u79ef\u3002",
  );
}

function buildTrainingSummaryPayload(job, metrics) {
  if (metrics) {
    return {
      summary: metrics.summary,
      progress: metrics.progress,
      kpis: metrics.kpis,
      data_quality_report: metrics.data_quality_report?.summary ?? metrics.data_quality_report,
    };
  }
  if (job?.error) {
    return { error: job.error, result: job.result, progress: job.progress };
  }
  if (job?.progress || job?.artifacts) {
    return { progress: job.progress, artifacts: job.artifacts };
  }
  return null;
}

function buildTrainingLogText(job, idleText) {
  const logs = Array.isArray(job?.logs) ? job.logs : [];
  return logs.length ? logs.slice(-160).join("\n") : idleText;
}

function clampUnit(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function firstDefinedMetric(...values) {
  for (const value of values) {
    if (typeof value === "number" && !Number.isNaN(value)) {
      return value;
    }
  }
  return null;
}

function pickPreferredMetricName(metrics, names) {
  if (!metrics || typeof metrics !== "object") {
    return null;
  }
  return (names ?? []).find((name) => typeof metrics[name] === "number") ?? null;
}
