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
