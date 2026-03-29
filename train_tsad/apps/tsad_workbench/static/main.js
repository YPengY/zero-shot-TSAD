/**
 * Browser entry point for the workbench UI.
 *
 * This file only wires shared chart helpers onto `window` for legacy callers
 * and then boots the main application module.
 */

import * as charts from "./charts.js";

window.TsadCharts = charts;
void import("./app.js");
