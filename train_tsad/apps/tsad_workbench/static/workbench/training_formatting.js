/**
 * Pure formatting and localized status text for the training monitor.
 */

/**
 * Build pure formatting helpers for training-monitor view models.
 */
export function createTrainingFormatting(deps) {
  const { getLocale, localizeText, metricLabels, t } = deps;

  function formatSplitLabel(split) {
    if (split === "train") {
      return localizeText("Train", "\u8bad\u7ec3");
    }
    if (split === "val") {
      return localizeText("Validation", "\u9a8c\u8bc1");
    }
    if (split === "test") {
      return localizeText("Test", "\u6d4b\u8bd5");
    }
    return String(split ?? "");
  }

  function formatTrainingStatus(status) {
    if (status === "running") {
      return localizeText("running", "\u8fd0\u884c\u4e2d");
    }
    if (status === "completed") {
      return localizeText("completed", "\u5df2\u5b8c\u6210");
    }
    if (status === "failed") {
      return localizeText("failed", "\u5931\u8d25");
    }
    if (status === "pending") {
      return localizeText("pending", "\u7b49\u5f85\u4e2d");
    }
    return String(status ?? "-");
  }

  /**
   * Describe the current training stage in user-facing prose.
   */
  function buildTrainingStageText(job, progress, kpis) {
    if (job?.error) {
      return localizeText(
        "Training failed. Inspect the raw details below.",
        "\u8bad\u7ec3\u5931\u8d25\uff0c\u8bf7\u5c55\u5f00\u4e0b\u65b9\u539f\u59cb\u660e\u7ec6\u6392\u67e5\u3002",
      );
    }

    const status = job?.status ?? progress?.status ?? kpis?.status ?? null;
    const stage = progress?.stage ?? kpis?.stage ?? null;
    const epochCurrent = progress?.epoch_current ?? kpis?.epoch_current ?? null;
    const epochTotal = progress?.epoch_total ?? kpis?.epoch_total ?? null;

    if (!status && !stage && epochCurrent == null) {
      return t("workbench.trainingIdle");
    }
    if (!job && !status && !stage && epochCurrent != null) {
      return localizeText(
        `Loaded saved metrics through epoch ${epochCurrent}${epochTotal != null ? `/${epochTotal}` : ""}.`,
        `\u5df2\u52a0\u8f7d\u5230\u7b2c ${epochCurrent}${epochTotal != null ? ` / ${epochTotal}` : ""} \u8f6e\u7684\u5386\u53f2\u6307\u6807\u3002`,
      );
    }
    if (status === "completed") {
      return localizeText(
        `Training completed at epoch ${epochCurrent ?? "-"}/${epochTotal ?? "-"}.`,
        `\u8bad\u7ec3\u5df2\u5b8c\u6210\uff0c\u7ed3\u675f\u4e8e\u7b2c ${epochCurrent ?? "-"} / ${epochTotal ?? "-"} \u8f6e\u3002`,
      );
    }
    if (stage === "validation") {
      return localizeText(
        `Running validation for epoch ${epochCurrent ?? "-"}/${epochTotal ?? "-"}.`,
        `\u6b63\u5728\u6267\u884c\u7b2c ${epochCurrent ?? "-"} / ${epochTotal ?? "-"} \u8f6e\u9a8c\u8bc1\u3002`,
      );
    }
    if (stage === "epoch_complete") {
      return localizeText(
        `Epoch ${epochCurrent ?? "-"}/${epochTotal ?? "-"} finished. Waiting for the next update.`,
        `\u7b2c ${epochCurrent ?? "-"} / ${epochTotal ?? "-"} \u8f6e\u5df2\u7ed3\u675f\uff0c\u7b49\u5f85\u4e0b\u4e00\u6b21\u66f4\u65b0\u3002`,
      );
    }
    if (stage === "starting" || status === "pending") {
      return localizeText(
        "Training job has started. Waiting for the first step.",
        "\u8bad\u7ec3\u4efb\u52a1\u5df2\u542f\u52a8\uff0c\u7b49\u5f85\u7b2c\u4e00\u6279\u8fdb\u5ea6\u3002",
      );
    }
    return localizeText(
      `Training epoch ${epochCurrent ?? "-"}/${epochTotal ?? "-"}.`,
      `\u6b63\u5728\u8bad\u7ec3\u7b2c ${epochCurrent ?? "-"} / ${epochTotal ?? "-"} \u8f6e\u3002`,
    );
  }

  function formatMetricLabel(metricName) {
    const rawName = String(metricName ?? "");
    const key = rawName.includes(".") ? rawName.split(".").slice(1).join(".") : rawName;
    return metricLabels[getLocale()]?.[key] ?? metricLabels.en[key] ?? key.replace(/_/g, " ");
  }

  function formatSplitMetricLabel(metricName) {
    const [split, rawName] = String(metricName ?? "").split(".", 2);
    if (!rawName) {
      return formatMetricLabel(split);
    }
    return `${formatSplitLabel(split)} ${formatMetricLabel(rawName)}`;
  }

  function formatPercent(value, digits = 1) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatSignedPercent(value, digits = 1) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    const sign = value > 0 ? "+" : "";
    return `${sign}${(value * 100).toFixed(digits)}pp`;
  }

  function formatLearningRate(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    if (Math.abs(value) >= 0.001) {
      return value.toFixed(5);
    }
    return value.toExponential(2);
  }

  function formatDuration(value) {
    if (typeof value !== "number" || Number.isNaN(value) || value < 0) {
      return "-";
    }
    const totalSeconds = Math.round(value);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return [hours, minutes, seconds]
      .map((part, index) => (index === 0 ? String(part) : String(part).padStart(2, "0")))
      .join(":");
  }

  return {
    buildTrainingStageText,
    formatDuration,
    formatLearningRate,
    formatMetricLabel,
    formatPercent,
    formatSignedPercent,
    formatSplitLabel,
    formatSplitMetricLabel,
    formatTrainingStatus,
  };
}
