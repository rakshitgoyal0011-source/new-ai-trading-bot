/* DALAL TERMINAL - frontend logic: live tape, command bar, renderers. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const out = $("output");
  const cmdInput = $("cmd");
  const SPARK = "▁▂▃▄▅▆▇█";

  let tapeEls = {};
  const cmdHistory = [];
  let histIdx = 0;

  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* ---- clock ---- */
  function tick() {
    $("clock").textContent = new Date().toTimeString().slice(0, 8);
  }
  setInterval(tick, 1000);
  tick();

  /* ---- health / mode badge ---- */
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      const el = $("mode");
      el.textContent = "MODE " + String(h.mode || "?").toUpperCase();
      if (h.mode === "live") el.classList.add("live");
    })
    .catch(() => {});

  /* ---- websocket tape ---- */
  function setConn(ok) {
    const el = $("conn");
    el.textContent = ok ? "LIVE" : "OFFLINE";
    el.className = "badge " + (ok ? "live" : "badge-dim");
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(proto + "://" + location.host + "/ws");
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 2000); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === "tape") onTape(msg.data);
    };
  }

  function buildTape(data) {
    const track = $("tape-track");
    track.innerHTML = "";
    tapeEls = {};
    for (let copy = 0; copy < 2; copy++) {
      data.forEach((row) => {
        const item = document.createElement("span");
        item.className = "tape-item";
        item.innerHTML =
          '<span class="sym">' + row.s + '</span> ' +
          '<span class="px"></span> <span class="ch"></span>';
        track.appendChild(item);
        (tapeEls[row.s] = tapeEls[row.s] || []).push(item);
      });
    }
  }

  function onTape(data) {
    if (!data || !data.length) return;
    if (Object.keys(tapeEls).length !== data.length) buildTape(data);
    data.forEach((row) => {
      const cls = row.d > 0 ? "up" : row.d < 0 ? "down" : "flat";
      const arrow = row.d > 0 ? "▲" : row.d < 0 ? "▼" : "—";
      (tapeEls[row.s] || []).forEach((item) => {
        const px = item.querySelector(".px");
        const ch = item.querySelector(".ch");
        px.textContent = row.ltp.toFixed(2);
        px.className = "px " + cls;
        ch.textContent = arrow + Math.abs(row.chg).toFixed(2) + "%";
        ch.className = "ch " + cls;
      });
    });
    updateSide(data);
  }

  function sideRow(r) {
    const cls = r.d > 0 ? "up" : r.d < 0 ? "down" : "flat";
    const sign = r.chg >= 0 ? "+" : "";
    return '<div class="side-row"><span class="sym">' + r.s + "</span>" +
      '<span class="' + cls + '">' + r.ltp.toFixed(2) + " " +
      sign + r.chg.toFixed(2) + "%</span></div>";
  }

  function updateSide(data) {
    $("watch").innerHTML = data.slice(0, 12).map(sideRow).join("");
    const sorted = data.slice().sort((a, b) => b.chg - a.chg);
    const movers = sorted.slice(0, 5).concat(sorted.slice(-5).reverse());
    $("movers").innerHTML = movers.map(sideRow).join("");
  }

  /* ---- renderers ---- */
  function sparkline(data) {
    if (!data || !data.length) return "";
    const min = Math.min.apply(null, data);
    const max = Math.max.apply(null, data);
    const span = max - min || 1;
    return data
      .map((v) => SPARK[Math.min(7, Math.floor(((v - min) / span) * 8))])
      .join("");
  }

  function toneOf(text) {
    const t = String(text).trim();
    if (/^\[\+\]/.test(t) || /bull|uptrend/i.test(t)) return "up";
    if (/^\[-\]/.test(t) || /bear|downtrend/i.test(t)) return "down";
    if (/^\+\d/.test(t)) return "up";
    if (/^-\d/.test(t)) return "down";
    return "";
  }

  function heatColor(score) {
    // 0 -> red, 50 -> amber, 100 -> green; smooth ramp
    const s = Math.max(0, Math.min(100, score));
    const r = s < 50 ? 255 : Math.round(255 - (s - 50) * 5.1);
    const g = s < 50 ? Math.round(s * 5.1) : 200;
    return "rgb(" + r + "," + g + ",40)";
  }

  function renderHeatmap(b) {
    const wrap = document.createElement("div");
    wrap.className = "heat-wrap";
    (b.sectors || []).forEach((sec) => {
      const row = document.createElement("div");
      row.className = "heat-row";
      const label = document.createElement("div");
      label.className = "heat-label";
      label.textContent = sec.name + "  avg " + sec.avg;
      row.appendChild(label);
      const cells = document.createElement("div");
      cells.className = "heat-cells";
      (sec.cells || []).forEach((c) => {
        const cell = document.createElement("div");
        cell.className = "heat-cell";
        cell.style.background = heatColor(c.score);
        cell.innerHTML =
          '<div class="heat-sym">' + esc(c.symbol) + "</div>" +
          '<div class="heat-score">' + c.score + "</div>";
        cell.title = c.symbol + " | composite " + c.score +
          " | chg " + c.change_pct + "%";
        cells.appendChild(cell);
      });
      row.appendChild(cells);
      wrap.appendChild(row);
    });
    return wrap;
  }

  function renderCandles(b) {
    const wrap = document.createElement("div");
    wrap.className = "candle-wrap";
    if (b.label) {
      const lab = document.createElement("div");
      lab.className = "spark-label";
      lab.textContent = b.label;
      wrap.appendChild(lab);
    }
    const data = b.ohlc || [];
    if (!data.length) return wrap;
    const W = 720, H = 220, padT = 6, padB = 12, padL = 36, padR = 6;
    const innerW = W - padL - padR;
    const innerH = H - padT - padB;
    const highs = data.map((d) => d[1]);
    const lows = data.map((d) => d[2]);
    const max = Math.max.apply(null, highs);
    const min = Math.min.apply(null, lows);
    const span = (max - min) || 1;
    const slot = innerW / data.length;
    const bodyW = Math.max(2, slot * 0.65);
    const y = (p) => padT + ((max - p) / span) * innerH;
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.classList.add("candle-svg");
    for (let i = 0; i < 4; i++) {
      const yy = padT + (i / 3) * innerH;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", padL); line.setAttribute("x2", W - padR);
      line.setAttribute("y1", yy); line.setAttribute("y2", yy);
      line.setAttribute("stroke", "#10171f");
      line.setAttribute("stroke-width", 0.5);
      svg.appendChild(line);
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", 2); label.setAttribute("y", yy + 3);
      label.setAttribute("fill", "#5f7180");
      label.setAttribute("font-size", 9);
      label.textContent = (max - (i / 3) * span).toFixed(2);
      svg.appendChild(label);
    }
    data.forEach((d, i) => {
      const o = d[0], h = d[1], l = d[2], c = d[3];
      const cx = padL + i * slot + slot / 2;
      const up = c >= o;
      const color = up ? "#2ecc71" : "#ff5247";
      const wick = document.createElementNS(svgNS, "line");
      wick.setAttribute("x1", cx); wick.setAttribute("x2", cx);
      wick.setAttribute("y1", y(h)); wick.setAttribute("y2", y(l));
      wick.setAttribute("stroke", color);
      wick.setAttribute("stroke-width", 1);
      svg.appendChild(wick);
      const body = document.createElementNS(svgNS, "rect");
      const top = Math.min(y(o), y(c));
      body.setAttribute("x", cx - bodyW / 2);
      body.setAttribute("y", top);
      body.setAttribute("width", bodyW);
      body.setAttribute("height", Math.max(1, Math.abs(y(o) - y(c))));
      body.setAttribute("fill", color);
      svg.appendChild(body);
    });
    wrap.appendChild(svg);
    return wrap;
  }

  function renderBlock(b) {
    const el = document.createElement("div");
    el.className = "blk";
    if (b.type === "text") {
      el.className += " blk-text";
      el.textContent = b.text;
    } else if (b.type === "note") {
      el.className += " blk-note";
      el.textContent = b.text;
    } else if (b.type === "disclaimer") {
      el.className += " blk-disclaimer";
      el.textContent = b.text;
    } else if (b.type === "keyval") {
      el.className += " keyval";
      el.innerHTML = (b.pairs || [])
        .map((p) => '<span class="kv"><span class="k">' + esc(p[0]) +
          '</span><span class="v ' + toneOf(p[1]) + '">' + esc(p[1]) +
          "</span></span>")
        .join("");
    } else if (b.type === "table") {
      const th = (b.headers || []).map((h) => "<th>" + esc(h) + "</th>").join("");
      const tr = (b.rows || [])
        .map((row) =>
          "<tr>" +
          row.map((c) => '<td class="' + toneOf(c) + '">' + esc(c) + "</td>").join("") +
          "</tr>")
        .join("");
      el.innerHTML = '<table class="blk-table"><thead><tr>' + th +
        "</tr></thead><tbody>" + tr + "</tbody></table>";
    } else if (b.type === "bars") {
      el.innerHTML = (b.items || [])
        .map((it) => {
          const v = Math.max(0, Math.min(100, it[1]));
          const color = v >= 60 ? "var(--up)" : v <= 40 ? "var(--down)" : "var(--amber)";
          return '<div class="bar-row"><span class="bar-label">' + esc(it[0]) +
            '</span><span class="bar-track"><span class="bar-fill" style="width:' +
            v + "%;background:" + color + '"></span></span>' +
            '<span class="bar-val">' + it[1] + "</span></div>";
        })
        .join("");
    } else if (b.type === "spark") {
      el.innerHTML = (b.label ? '<div class="spark-label">' + esc(b.label) +
        "</div>" : "") + '<div class="spark">' + sparkline(b.data) + "</div>";
    } else if (b.type === "candles") {
      el.appendChild(renderCandles(b));
    } else if (b.type === "heatmap") {
      el.appendChild(renderHeatmap(b));
    } else {
      el.className += " blk-text";
      el.textContent = JSON.stringify(b);
    }
    return el;
  }

  function renderResponse(resp) {
    out.innerHTML = "";
    const title = document.createElement("div");
    title.className = "result-title";
    title.innerHTML = esc(resp.title || "RESULT") +
      (resp.subtitle ? ' <span class="result-sub">- ' + esc(resp.subtitle) +
        "</span>" : "");
    out.appendChild(title);
    (resp.blocks || []).forEach((b) => out.appendChild(renderBlock(b)));
    out.scrollTop = 0;
  }

  /* ---- command bar ---- */
  async function runCommand(cmd) {
    out.innerHTML = '<div class="muted">running &nbsp;' + esc(cmd) + ' &hellip;</div>';
    try {
      const res = await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
      renderResponse(await res.json());
    } catch (err) {
      out.innerHTML = '<div class="blk blk-note">request failed: ' +
        esc(String(err)) + "</div>";
    }
  }

  cmdInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const cmd = cmdInput.value.trim();
      if (!cmd) return;
      cmdHistory.push(cmd);
      histIdx = cmdHistory.length;
      cmdInput.value = "";
      runCommand(cmd);
    } else if (e.key === "ArrowUp") {
      if (histIdx > 0) cmdInput.value = cmdHistory[--histIdx];
      e.preventDefault();
    } else if (e.key === "ArrowDown") {
      if (histIdx < cmdHistory.length - 1) cmdInput.value = cmdHistory[++histIdx];
      else { histIdx = cmdHistory.length; cmdInput.value = ""; }
      e.preventDefault();
    }
  });

  document.addEventListener("click", () => {
    if (!window.getSelection().toString()) cmdInput.focus();
  });

  /* ---- boot ---- */
  connect();
  runCommand("HELP");
  cmdInput.focus();
})();
