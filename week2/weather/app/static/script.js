// Validated categorical palette, fixed slot order (dataviz reference palette)
const PARAMS = [
  { key: "temperature", color: "#2a78d6", type: "scatter" },
  { key: "humidity", color: "#eb6834", type: "scatter" },
  { key: "pressure", color: "#1baf7a", type: "scatter" },
  { key: "wind_speed", color: "#eda100", type: "scatter" },
  { key: "precipitation", color: "#e87ba4", type: "bar" },
];

let currentLang = "en";
let currentCity = "";
let translations = null;
let forecastDays = [];

// navigator.languages mirrors the browser's Accept-Language preference list
function pickLanguage(available) {
  const saved = localStorage.getItem("lang");
  if (available[saved]) return saved;
  for (const code of navigator.languages || [navigator.language]) {
    let prefix = code.slice(0, 2).toLowerCase();
    if (prefix === "kz") prefix = "kk";
    if (available[prefix]) return prefix;
  }
  return "en";
}

function setLanguage(lang) {
  currentLang = lang;
  localStorage.setItem("lang", lang);
  $.getJSON(`/i18n/${lang}.json`, function (t) {
    translations = t;
    renderAll();
  });
}

function setCity(city) {
  currentCity = city;
  localStorage.setItem("city", city);
  $.getJSON("/api/forecast", { city: city }, function (days) {
    forecastDays = days;
    renderAll();
  });
}

function renderAll() {
  // Language and forecast load in parallel; render when both have arrived
  if (!translations || !forecastDays.length) return;
  const t = translations;
  document.documentElement.lang = currentLang;
  document.title = `${currentCity} · ${t.title}`;
  $("#title").text(`${currentCity} · ${t.title}`);
  $("#subtitle").text(t.subtitle);
  $("#model-link").text(t.model_link || "Model verification");
  $("#readme-link").text(t.readme_link || "Project docs");
  $("#forecast-table").empty();
  $("#charts").empty();
  renderTable(t);
  PARAMS.forEach((p) => renderChart(t, p));
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(currentLang, { day: "numeric", month: "short" });
}

function renderTable(t) {
  const head = [t.date].concat(PARAMS.map((p) => `${t[p.key]}, ${t.units[p.key]}`));
  const $table = $("#forecast-table");
  $table.append($("<tr>").append(head.map((h) => $("<th>").text(h))));
  forecastDays.forEach((day) => {
    const cells = [$("<td>").addClass("date").text(formatDate(day.date))].concat(
      PARAMS.map((p) => $("<td>").text(day[p.key]))
    );
    $table.append($("<tr>").append(cells));
  });
}

function renderChart(t, param) {
  const container = $("<div>").attr("id", `chart-${param.key}`).appendTo("#charts");
  const trace = {
    type: param.type,
    x: forecastDays.map((d) => d.date),
    y: forecastDays.map((d) => d[param.key]),
    line: { color: param.color, width: 2 },
    marker: { color: param.color },
    hovertemplate: `%{y} ${t.units[param.key]}<extra>%{x}</extra>`,
  };
  const layout = {
    title: { text: `${t[param.key]}, ${t.units[param.key]}`, font: { size: 14, color: "#52514e" } },
    height: 240,
    margin: { l: 45, r: 15, t: 40, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#898781", size: 11, family: "system-ui, sans-serif" },
    xaxis: { gridcolor: "#e1e0d9", linecolor: "#c3c2b7" },
    yaxis: { gridcolor: "#e1e0d9", zeroline: false },
  };
  Plotly.newPlot(container[0], [trace], layout, { displayModeBar: false, responsive: true });
}

$("#language").on("change", function () {
  setLanguage(this.value);
});

$("#city").on("change", function () {
  setCity(this.value);
});

$.getJSON("/api/languages", function (languages) {
  const $select = $("#language");
  for (const [code, name] of Object.entries(languages)) {
    $select.append($("<option>").val(code).text(name));
  }
  const lang = pickLanguage(languages);
  $select.val(lang);
  setLanguage(lang);
});

$.getJSON("/api/cities", function (cities) {
  const names = Object.keys(cities);
  const $select = $("#city");
  names.forEach((name) => $select.append($("<option>").val(name).text(name)));
  const saved = localStorage.getItem("city");
  const city = names.includes(saved) ? saved : names.includes("Moscow") ? "Moscow" : names[0];
  $select.val(city);
  setCity(city);
});
