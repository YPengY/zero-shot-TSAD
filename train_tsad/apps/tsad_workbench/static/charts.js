/**
 * Shared canvas and SVG renderers for preview and workbench surfaces.
 *
 * These helpers stay presentation-only: they consume already prepared series,
 * masks, and metric arrays without performing data fetching or state updates.
 */

export const palette = [
  "#b55232",
  "#28536b",
  "#7d5a50",
  "#5f0f40",
  "#4f772d",
  "#6f1d1b",
  "#4361ee",
  "#2a9d8f",
  "#8c5e58",
  "#f4a261",
];

export function getNodeColor(nodeIndex) {
  const normalized = Number(nodeIndex);
  if (!Number.isFinite(normalized)) {
    return palette[0];
  }
  const paletteIndex = ((Math.trunc(normalized) % palette.length) + palette.length) % palette.length;
  return palette[paletteIndex];
}

export function rememberCanvasLogicalSize(canvas) {
  if (!canvas) {
    return;
  }
  if (!canvas.dataset.logicalWidth) {
    canvas.dataset.logicalWidth = String(canvas.getAttribute("width") || canvas.width || 300);
  }
  if (!canvas.dataset.logicalHeight) {
    canvas.dataset.logicalHeight = String(canvas.getAttribute("height") || canvas.height || 150);
  }
}

/**
 * Resize a canvas to match CSS pixels while keeping high-DPI rendering sharp.
 */
export function prepareCanvas2d(canvas) {
  rememberCanvasLogicalSize(canvas);
  const fallbackWidth = Math.max(1, Number(canvas.dataset.logicalWidth || 300));
  const fallbackHeight = Math.max(1, Number(canvas.dataset.logicalHeight || 150));
  const rect = canvas.getBoundingClientRect();
  const cssWidth = rect.width > 0 ? rect.width : fallbackWidth;
  const cssHeight = rect.height > 0 ? rect.height : fallbackHeight;
  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const displayWidth = Math.max(1, Math.round(cssWidth * pixelRatio));
  const displayHeight = Math.max(1, Math.round(cssHeight * pixelRatio));
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.imageSmoothingEnabled = true;
  return { context, width: cssWidth, height: cssHeight, pixelRatio };
}

export function clearCanvas(canvas) {
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
}

/**
 * Draw the multi-node preview chart used in the studio view.
 *
 * `data` is expected to be `[T][D]`, and `maskAny` is the collapsed
 * per-timestep anomaly mask used to shade anomalous regions.
 */
export function drawLineChart(canvas, data, selectedNodes, maskAny, options = {}) {
  const { highlightedEvent = null } = options;
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  if (!data || selectedNodes.length === 0) {
    return;
  }

  const padding = { left: 60, right: 24, top: 24, bottom: 40 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = selectedNodes
    .flatMap((node) => data.map((row) => row[node]))
    .filter((value) => Number.isFinite(value));
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    minValue = -1;
    maxValue = 1;
  } else if (Math.abs(maxValue - minValue) < 1e-8) {
    minValue -= 1;
    maxValue += 1;
  }

  context.fillStyle = "rgba(181, 82, 50, 0.08)";
  let inMask = false;
  let maskStart = 0;
  maskAny.forEach((value, index) => {
    if (value && !inMask) {
      inMask = true;
      maskStart = index;
    }
    const shouldClose = !value && inMask;
    const isLast = inMask && index === maskAny.length - 1;
    if (shouldClose || isLast) {
      const maskEnd = shouldClose ? index : index + 1;
      const x1 = padding.left + (maskStart / Math.max(1, data.length - 1)) * plotWidth;
      const x2 = padding.left + ((maskEnd - 1) / Math.max(1, data.length - 1)) * plotWidth;
      context.fillRect(x1, padding.top, Math.max(2, x2 - x1), plotHeight);
      inMask = false;
    }
  });

  if (highlightedEvent && Number.isFinite(highlightedEvent.t_start) && Number.isFinite(highlightedEvent.t_end)) {
    const start = Math.max(0, Math.min(data.length - 1, Number(highlightedEvent.t_start)));
    const end = Math.max(start + 1, Math.min(data.length, Number(highlightedEvent.t_end)));
    const x1 = padding.left + (start / Math.max(1, data.length - 1)) * plotWidth;
    const x2 = padding.left + ((Math.max(start, end - 1)) / Math.max(1, data.length - 1)) * plotWidth;
    context.fillStyle = "rgba(40, 83, 107, 0.16)";
    context.fillRect(x1, padding.top, Math.max(3, x2 - x1), plotHeight);
    context.strokeStyle = "rgba(40, 83, 107, 0.7)";
    context.lineWidth = 1.4;
    context.beginPath();
    context.moveTo(x1, padding.top);
    context.lineTo(x1, padding.top + plotHeight);
    context.moveTo(x2, padding.top);
    context.lineTo(x2, padding.top + plotHeight);
    context.stroke();
  }

  drawAxes(context, padding, width, height, minValue, maxValue, data.length);
  selectedNodes.forEach((node) => {
    context.strokeStyle = getNodeColor(node);
    context.lineWidth = 2;
    context.beginPath();
    data.forEach((row, step) => {
      const x = padding.left + (step / Math.max(1, data.length - 1)) * plotWidth;
      const ratio = (row[node] - minValue) / (maxValue - minValue);
      const y = padding.top + plotHeight - ratio * plotHeight;
      if (step === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  });
}

/**
 * Draw a node-by-time anomaly mask view for the selected nodes.
 */
export function drawMaskHeatmap(canvas, pointMask, selectedNodes, options = {}) {
  const {
    xStart = 0,
    xEnd = Array.isArray(pointMask) ? pointMask.length : 0,
    emptyMessage = "No mask data available.",
    nodeLabel = "node",
  } = options;
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  if (!Array.isArray(pointMask) || pointMask.length === 0 || selectedNodes.length === 0) {
    context.fillStyle = "#6f6559";
    context.font = "14px Segoe UI";
    context.textAlign = "center";
    context.fillText(emptyMessage, width / 2, height / 2);
    context.textAlign = "start";
    return;
  }

  const padding = { left: 90, right: 20, top: 20, bottom: 34 };
  const rows = selectedNodes.length;
  const columns = pointMask.length;
  const cellWidth = (width - padding.left - padding.right) / Math.max(1, columns);
  const cellHeight = (height - padding.top - padding.bottom) / Math.max(1, rows);
  selectedNodes.forEach((node, rowIndex) => {
    context.fillStyle = "#1e1a16";
    context.font = "12px Segoe UI";
    context.fillText(`${nodeLabel} ${node}`, 16, padding.top + rowIndex * cellHeight + cellHeight * 0.7);
    for (let column = 0; column < columns; column += 1) {
      context.fillStyle = pointMask[column]?.[node] ? "#b55232" : "#f3ebe0";
      context.fillRect(
        padding.left + column * cellWidth,
        padding.top + rowIndex * cellHeight,
        Math.max(1, cellWidth),
        Math.max(1, cellHeight - 1),
      );
    }
  });
  context.fillStyle = "#6f6559";
  context.font = "12px Segoe UI";
  context.fillText(`t=${Math.max(0, xStart)}`, padding.left, height - 10);
  const endLabel = `t=${Math.max(xStart, xEnd - 1)}`;
  const endWidth = context.measureText(endLabel).width;
  context.fillText(endLabel, Math.max(padding.left + 20, width - padding.right - endWidth), height - 10);
}

/**
 * Render the sampled DAG with parent-to-child edges.
 */
export function drawDag(svg, parents, topoOrder, selectedNodes) {
  svg.innerHTML = "";
  if (!parents) {
    return;
  }

  const width = 980;
  const height = 420;
  const depths = Array.from({ length: parents.length }, () => 0);
  topoOrder.forEach((node) => {
    if (parents[node].length > 0) {
      depths[node] = Math.max(...parents[node].map((parent) => depths[parent] + 1));
    }
  });

  const levels = new Map();
  topoOrder.forEach((node) => {
    const depth = depths[node];
    if (!levels.has(depth)) {
      levels.set(depth, []);
    }
    levels.get(depth).push(node);
  });

  const maxDepth = Math.max(...levels.keys(), 0);
  const positions = new Map();
  levels.forEach((nodes, depth) => {
    nodes.forEach((node, index) => {
      const x = maxDepth === 0 ? width / 2 : 90 + (depth / maxDepth) * (width - 180);
      const y = nodes.length === 1 ? height / 2 : 60 + (index / (nodes.length - 1)) * (height - 120);
      positions.set(node, { x, y });
    });
  });

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5a554e"></path>
    </marker>
  `;
  svg.appendChild(defs);

  parents.forEach((nodeParents, child) => {
    nodeParents.forEach((parent) => {
      const from = positions.get(parent);
      const to = positions.get(child);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", from.x + 22);
      line.setAttribute("y1", from.y);
      line.setAttribute("x2", to.x - 22);
      line.setAttribute("y2", to.y);
      line.setAttribute("stroke", "#5a554e");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("marker-end", "url(#arrow)");
      svg.appendChild(line);
    });
  });

  positions.forEach(({ x, y }, node) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 20);
    circle.setAttribute("fill", selectedNodes.includes(node) ? getNodeColor(node) : "#8e857a");
    circle.setAttribute("stroke", "#1e1a16");
    circle.setAttribute("stroke-width", "1.5");
    svg.appendChild(circle);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y + 5);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#ffffff");
    text.setAttribute("font-size", "12");
    text.setAttribute("font-weight", "700");
    text.textContent = String(node);
    svg.appendChild(text);
  });
}

/**
 * Compare one observed feature series against its optional reference series.
 *
 * `mask` highlights anomaly regions, while `paddingMask` marks right-padded
 * timesteps in packed-window previews.
 */
export function drawSeriesPairChart(canvas, primary, reference, mask, paddingMask) {
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  if (!primary?.length) {
    return;
  }
  const padding = { left: 56, right: 20, top: 18, bottom: 34 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = [...primary, ...(reference ?? [])].filter((value) => Number.isFinite(value));
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || Math.abs(maxValue - minValue) < 1e-8) {
    minValue = -1;
    maxValue = 1;
  }
  if (mask?.length) {
    context.fillStyle = "rgba(181, 82, 50, 0.08)";
    mask.forEach((value, index) => {
      if (!value) {
        return;
      }
      const x = padding.left + (index / Math.max(1, primary.length - 1)) * plotWidth;
      context.fillRect(x, padding.top, Math.max(2, plotWidth / Math.max(primary.length, 1)), plotHeight);
    });
  }
  if (paddingMask?.length) {
    context.fillStyle = "rgba(40, 83, 107, 0.08)";
    paddingMask.forEach((value, index) => {
      if (!value) {
        return;
      }
      const x = padding.left + (index / Math.max(1, primary.length - 1)) * plotWidth;
      context.fillRect(x, padding.top, Math.max(2, plotWidth / Math.max(primary.length, 1)), plotHeight);
    });
  }
  drawAxes(context, padding, width, height, minValue, maxValue, primary.length);
  drawSimpleLine(context, primary, padding, plotWidth, plotHeight, minValue, maxValue, "#b55232", 2.2);
  if (reference?.length) {
    drawSimpleLine(context, reference, padding, plotWidth, plotHeight, minValue, maxValue, "#28536b", 1.8);
  }
}

/**
 * Render patch labels as a compact binary bar strip.
 */
export function drawPatchBarChart(canvas, patchLabels) {
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  if (!patchLabels?.length) {
    return;
  }
  const padding = { left: 20, right: 20, top: 30, bottom: 24 };
  const barWidth = (width - padding.left - padding.right) / patchLabels.length;
  patchLabels.forEach((value, index) => {
    context.fillStyle = value ? "#b55232" : "#d8cdbd";
    context.fillRect(
      padding.left + index * barWidth,
      padding.top,
      Math.max(2, barWidth - 1),
      height - padding.top - padding.bottom,
    );
  });
}

/**
 * Draw one or more training-history curves over epoch index.
 */
export function drawMetricChart(canvas, epochs, seriesMap, metricNames) {
  const { context, width, height } = prepareCanvas2d(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  const activeNames = (metricNames ?? []).filter((name) => Array.isArray(seriesMap?.[name]));
  if (!activeNames.length) {
    return;
  }
  const values = activeNames.flatMap((name) => seriesMap[name].filter((value) => Number.isFinite(value)));
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || Math.abs(maxValue - minValue) < 1e-8) {
    minValue = 0;
    maxValue = 1;
  }
  const padding = { left: 56, right: 20, top: 18, bottom: 34 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  drawAxes(context, padding, width, height, minValue, maxValue, Math.max(epochs.length, 1), {
    xStartLabel: epochs.length ? `e${epochs[0]}` : "e1",
    xEndLabel: epochs.length ? `e${epochs[epochs.length - 1]}` : "e1",
  });
  activeNames.forEach((name, index) => {
    const valuesForLine = seriesMap[name].map((value) => (Number.isFinite(value) ? value : null));
    context.strokeStyle = palette[index % palette.length];
    context.lineWidth = 2;
    let previousPoint = null;
    valuesForLine.forEach((value, pointIndex) => {
      if (value == null) {
        previousPoint = null;
        return;
      }
      const x = padding.left + (pointIndex / Math.max(1, valuesForLine.length - 1)) * plotWidth;
      const ratio = (value - minValue) / Math.max(1e-8, maxValue - minValue);
      const y = padding.top + plotHeight - ratio * plotHeight;
      if (previousPoint == null) {
        context.beginPath();
        context.moveTo(x, y);
      } else {
        context.beginPath();
        context.moveTo(previousPoint.x, previousPoint.y);
        context.lineTo(x, y);
        context.stroke();
      }
      context.fillStyle = palette[index % palette.length];
      context.beginPath();
      context.arc(x, y, 2.6, 0, Math.PI * 2);
      context.fill();
      previousPoint = { x, y };
    });
  });
}

function drawAxes(context, padding, width, height, minValue, maxValue, length, options = {}) {
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  context.strokeStyle = "#bcb3a6";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();
  context.fillStyle = "#6f6559";
  context.font = "12px Segoe UI";
  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = padding.top + plotHeight - ratio * plotHeight;
    const value = minValue + ratio * (maxValue - minValue);
    context.fillText(value.toFixed(2), 6, y + 4);
    context.strokeStyle = "rgba(111, 101, 89, 0.15)";
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
  }
  const xStartLabel = options.xStartLabel ?? "t=0";
  const xEndLabel = options.xEndLabel ?? `t=${Math.max(length - 1, 0)}`;
  context.fillText(String(xStartLabel), padding.left, height - 10);
  const endText = String(xEndLabel);
  const endWidth = context.measureText(endText).width;
  context.fillText(endText, Math.max(padding.left + 20, width - padding.right - endWidth), height - 10);
}

function drawSimpleLine(context, values, padding, plotWidth, plotHeight, minValue, maxValue, color, width) {
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  values.forEach((value, index) => {
    const x = padding.left + (index / Math.max(1, values.length - 1)) * plotWidth;
    const ratio = (value - minValue) / Math.max(1e-8, maxValue - minValue);
    const y = padding.top + plotHeight - ratio * plotHeight;
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();
}
