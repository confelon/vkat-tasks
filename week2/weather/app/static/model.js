// Technical verification page — English only, units as stored in training data
const TARGETS = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"];
const UNITS = { temperature: "°C", humidity: "%", pressure: "hPa", wind_speed: "m/s", precipitation: "mm" };
const LABELS = {
  climatology: "Climatology (seasonal mean)",
  dummy: "Dummy (overall mean)",
  ridge: "Ridge (linear)",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
};

function renderLadder(metrics) {
  const $table = $("#ladder").empty();
  const head = ["Model"].concat(TARGETS.map((t) => `${t.replace("_", " ")}, ${UNITS[t]}`));
  $table.append($("<tr>").append(head.map((h) => $("<th>").text(h))));

  const rows = { climatology: metrics.climatology, ...metrics.models };
  for (const [name, report] of Object.entries(rows)) {
    const label = (name === metrics.winner ? "★ " : "") + (LABELS[name] || name);
    const cells = [$("<td>").addClass("date").text(label)];
    TARGETS.forEach((target) => {
      let text = report.mae[target].toFixed(2);
      if (report.skill && report.skill[target] !== undefined) {
        text += ` (${report.skill[target] >= 0 ? "+" : ""}${report.skill[target].toFixed(2)})`;
      }
      const bias = report.bias[target];
      cells.push(
        $("<td>")
          .text(text)
          .append("<br>", $("<small>").addClass("bias").text(`bias ${bias >= 0 ? "+" : ""}${bias.toFixed(2)}`))
      );
    });
    $table.append($("<tr>").append(cells));
  }
}

function renderTraining(metrics) {
  const t = metrics.training;
  const rows = [
    ["Training period", `${t.first_day} → ${t.last_day}`],
    ["Training samples", `${t.train_samples} days`],
    ["Validation window", `last ${t.validation_days} days (${t.validation_samples} forecasts)`],
    ["Forecast horizon", `${t.horizon_days} days ahead`],
    ["Features per sample", t.features],
    ["Model outputs", `${t.outputs} (5 values × ${t.horizon_days} days)`],
    ["Sample weighting", `linear, ${t.oldest_sample_weight} (oldest) → 1.0 (newest)`],
  ];
  const $table = $("#training").empty();
  rows.forEach(([label, value]) =>
    $table.append($("<tr>").append($("<td>").addClass("date").text(label), $("<td>").text(value)))
  );
}

function renderHorizonChart(metrics) {
  const curve = metrics.temperature_mae_by_horizon;
  const trace = {
    type: "scatter",
    x: curve.map((_, i) => i + 1),
    y: curve,
    line: { color: "#2a78d6", width: 2 },
    hovertemplate: "+%{x} days: %{y:.2f} °C<extra></extra>",
  };
  const layout = {
    title: { text: "Temperature MAE by horizon (°C) — why forecasts degrade", font: { size: 14, color: "#52514e" } },
    height: 280,
    margin: { l: 45, r: 15, t: 40, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#898781", size: 11, family: "system-ui, sans-serif" },
    xaxis: { title: { text: "days ahead" }, gridcolor: "#e1e0d9", linecolor: "#c3c2b7", dtick: 5 },
    yaxis: { gridcolor: "#e1e0d9", zeroline: false, rangemode: "tozero" },
  };
  Plotly.newPlot("horizon-chart", [trace], layout, { displayModeBar: false, responsive: true });
}

function loadCity(city) {
  localStorage.setItem("city", city);
  $.getJSON("/api/models", { city: city }, function (byModel) {
    const metrics = Object.values(byModel)[0];
    renderLadder(metrics);
    $("#summary").text(`Winner: ${LABELS[metrics.winner] || metrics.winner} — saved and used for the forecast page.`);
    renderTraining(metrics);
    renderHorizonChart(metrics);
  });
}

$("#city").on("change", function () {
  loadCity(this.value);
});

$.getJSON("/api/cities", function (cities) {
  const names = Object.keys(cities);
  const $select = $("#city");
  names.forEach((name) => $select.append($("<option>").val(name).text(name)));
  const saved = localStorage.getItem("city");
  const city = names.includes(saved) ? saved : names.includes("Moscow") ? "Moscow" : names[0];
  $select.val(city);
  loadCity(city);
});
