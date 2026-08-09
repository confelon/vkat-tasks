// Fleet page: renders the offline benchmark reports served from app/data/*.json.
// The "Not generated yet" hints in the HTML stay visible until data arrives.
// The three reports live in tabs; the chosen tab is remembered in localStorage.

$(".tab").on("click", function () {
  $(".tab").removeClass("active");
  $(this).addClass("active");
  $(".tab-pane").removeClass("active");
  $("#" + this.dataset.pane).addClass("active");
  localStorage.setItem("fleetTab", this.dataset.pane);
  // Plotly sizes charts to 0 width inside hidden panes; a resize event fixes them once visible
  window.dispatchEvent(new Event("resize"));
});

const $savedTab = $(`.tab[data-pane="${localStorage.getItem("fleetTab")}"]`);
($savedTab.length ? $savedTab : $(".tab").first()).trigger("click");

function signed(value, digits) {
  return (value >= 0 ? "+" : "") + value.toFixed(digits);
}

// Every table here is a small grid: click a header to sort by that column,
// click again to flip the direction; ▲/▼ marks the active column.
function fillTable($table, headers, rows) {
  $table.data({ rows: rows, sortColumn: null, sortAsc: true });
  $table.empty().append(
    $("<tr>").append(
      headers.map((h, i) => $("<th>").addClass("sortable").text(h).on("click", () => sortTable($table, i)))
    )
  );
  renderRows($table, rows);
}

function renderRows($table, rows) {
  $table.find("tr").slice(1).remove();
  rows.forEach((cells) => {
    $table.append(
      $("<tr>").append(cells.map((value, i) => $("<td>").toggleClass("date", i === 0).text(value)))
    );
  });
}

function sortTable($table, column) {
  const asc = $table.data("sortColumn") === column ? !$table.data("sortAsc") : true;
  $table.data({ sortColumn: column, sortAsc: asc });
  $table.find("th").each(function (i) {
    const label = $(this).text().replace(/ [▲▼]$/, "");
    $(this).text(i === column ? `${label} ${asc ? "▲" : "▼"}` : label);
  });
  const sorted = [...$table.data("rows")].sort((a, b) => compareCells(a[column], b[column]) * (asc ? 1 : -1));
  renderRows($table, sorted);
}

// Cells like "2.58 (+0.12)" or "10y" sort by their leading number, the rest as text
function compareCells(a, b) {
  const numA = parseFloat(a);
  const numB = parseFloat(b);
  if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
  return String(a).localeCompare(String(b));
}

function fillAnomalies($container, anomalies) {
  $container.empty().append($("<h2>").text(`Anomalies (${anomalies.length})`));
  if (!anomalies.length) {
    $container.append($("<p>").addClass("note").text("No anomalies detected."));
  }
  anomalies.forEach((flag) => $container.append($("<p>").addClass("note").text("⚠ " + flag)));
}

function chartLayout(title, extraYAxis) {
  return {
    title: { text: title, font: { size: 13, color: "#52514e" } },
    height: 220,
    margin: { l: 40, r: 10, t: 35, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#898781", size: 11, family: "system-ui, sans-serif" },
    xaxis: { gridcolor: "#e1e0d9", linecolor: "#c3c2b7", type: "category" },
    yaxis: { gridcolor: "#e1e0d9", zeroline: false, ...extraYAxis },
  };
}

// Three "why this variant won" charts per experiment, all from report.aggregate.
// Colors follow the validated dataviz categorical palette in fixed slot order.
function renderExperimentCharts(prefix, report, label) {
  const variants = report.aggregate.map((r) => label(r.variant));
  const options = { displayModeBar: false, responsive: true };
  Plotly.newPlot(`${prefix}-votes`, [{
    type: "bar",
    x: variants,
    y: report.aggregate.map((r) => r.best_in_cities),
    marker: { color: "#2a78d6" },
    hovertemplate: "best in %{y} cities<extra>%{x}</extra>",
  }], chartLayout("The vote: best variant, city count"), options);
  // Zoom the y-axis to 5× the data span (data swings over ~20% of the plot):
  // anchored at zero this line looks perfectly flat and says nothing
  const maes = report.aggregate.map((r) => r.mean_temp_mae);
  const center = (Math.min(...maes) + Math.max(...maes)) / 2;
  const half = (Math.max(...maes) - Math.min(...maes)) * 2.5;
  Plotly.newPlot(`${prefix}-mae`, [{
    type: "scatter",
    x: variants,
    y: maes,
    line: { color: "#eb6834", width: 2 },
    marker: { color: "#eb6834", size: 8 },
    hovertemplate: "%{y:.3f} °C<extra>%{x}</extra>",
  }], chartLayout("Accuracy: mean temp MAE, °C (zoomed)", { range: [center - half, center + half] }), options);
  Plotly.newPlot(`${prefix}-bias`, [{
    type: "scatter",
    x: variants,
    y: report.aggregate.map((r) => r.mean_temp_bias),
    line: { color: "#1baf7a", width: 2 },
    marker: { color: "#1baf7a", size: 8 },
    hovertemplate: "%{y:+.2f} °C<extra>%{x}</extra>",
  }], chartLayout("The cost: mean temp bias, °C", { zeroline: true, zerolinecolor: "#c3c2b7" }), options);
}

// Both fleet experiments (depth, weight) share one JSON shape and one renderer;
// `label` formats a variant for display (10 -> "10y", 0.2 -> "w=0.2").
function renderExperiment(prefix, url, variantHeader, label) {
  $.getJSON(url, function (report) {
    renderExperimentCharts(prefix, report, label);
    $(`#${prefix}-meta`).text(
      `Generated on ${report.generated_at} for ${Object.keys(report.cities).length} cities — ${report.description}.`
    );
    fillTable(
      $(`#${prefix}-aggregate`),
      [variantHeader, "Mean temp MAE, °C", "Median temp MAE, °C", "Mean temp bias, °C", "Mean overall", "Best in cities"],
      report.aggregate.map((r) => [
        label(r.variant), r.mean_temp_mae.toFixed(3), r.median_temp_mae.toFixed(3),
        signed(r.mean_temp_bias, 2), r.mean_overall.toFixed(3), r.best_in_cities,
      ])
    );
    fillTable(
      $(`#${prefix}-table`),
      ["City"].concat(report.variants.map((v) => `${label(v)} MAE (bias)`)),
      Object.entries(report.cities).map(([city, rows]) => [
        city,
        ...rows.map((r) => `${r.temp_mae.toFixed(2)} (${signed(r.temp_bias, 2)})`),
      ])
    );
    fillAnomalies($(`#${prefix}-anomalies`), report.anomalies);
  });
}

$.getJSON("/api/benchmark", function (report) {
  $("#benchmark-meta").text(`Generated on ${report.generated_at} from ${report.models} trained model(s). Sorted by overall score (MAE / std, lower = better).`);
  fillTable(
    $("#benchmark-table"),
    ["City", "Model", "Temp MAE, °C", "Temp bias, °C", "Temp skill", "Overall"],
    report.fleet.map((r) => [
      r.city, r.model,
      r.temp_mae.toFixed(2), signed(r.temp_bias, 2), signed(r.temp_skill, 3), r.overall.toFixed(3),
    ])
  );
  fillAnomalies($("#benchmark-anomalies"), report.anomalies);
});

renderExperiment("depth", "/api/depth-benchmark", "Depth", (v) => `${v}y`);
renderExperiment("weight", "/api/weight-benchmark", "Weight", (v) => `w=${v}`);
