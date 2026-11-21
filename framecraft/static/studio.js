const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fileUrl = (job, path) =>
  "/api/jobs/" +
  job +
  "/files/" +
  path.split("/").map(encodeURIComponent).join("/");
const label = {
  queued: "Sırada",
  running: "Çalışıyor",
  complete: "Tamamlandı",
  needs_review: "İnceleme gerekli",
  failed: "Hata",
  interrupted: "Kesildi",
  create: "Kreatif",
  reference: "Referans",
  install: "Kurulum",
  compare: "Karşılaştırma",
};
const fields = {
  TIKTOK_CLIENT_ID: "TikTok uygulama ID",
  TIKTOK_CLIENT_SECRET: "TikTok uygulama sırrı",
  META_CLIENT_ID: "Meta uygulama ID",
  META_CLIENT_SECRET: "Meta uygulama sırrı",
  COMFY_URL: "ComfyUI adresi",
  COMFY_MODELS: "ComfyUI models klasörü",
  SD_HOME: "Bağımsız H3 klasörü",
  LLM_URL: "Dil ve görsel model API adresi",
  LLM_MODEL: "Yönetmen modeli",
  VISION_MODEL: "Görsel analiz modeli",
  H3_LORA: "Ek H3 LoRA dosyası",
  TRANSCRIBE_MODEL: "Yerel Whisper modeli (isteğe bağlı)",
  VOICE_PYTHON: "FreyaTTS Python yolu",
  FREYA_SOURCE: "FreyaTTS kaynak klasörü",
  FREYA_MODEL: "FreyaTTS ağırlıkları",
  VOICE_DEVICE: "FreyaTTS cihazı (cpu / cuda)",
  META_PAGE: "Facebook sayfa ID",
  META_AD_ACCOUNT: "Meta reklam hesabı ID",
  LLM_KEY: "Model API anahtarı",
  CIVITAI_TOKEN: "Civitai token",
  TIKTOK_TOKEN: "TikTok kullanıcı token · video.publish",
  META_TOKEN: "Meta erişim token",
};
function note(s) {
  $("#notice").hidden = !s;
  $("#notice").textContent = s;
}
async function api(path, method = "GET", data) {
  const opt = { method, headers: { "X-Framecraft": "1" } };
  if (data instanceof FormData) opt.body = data;
  else if (data !== undefined) {
    opt.body = JSON.stringify(data);
    opt.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(path, opt),
    body = await r.json();
  if (!r.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail || body),
    );
  return body;
}
const goJob = (r) => (location.hash = "job/" + r.id);
async function form(formId, fn) {
  $(formId).addEventListener("submit", async (e) => {
    e.preventDefault();
    const b = e.target.querySelector("button");
    b.disabled = true;
    note("");
    try {
      await fn(new FormData(e.target));
    } catch (err) {
      note(err.message);
    } finally {
      b.disabled = false;
    }
  });
}
form("#productForm", async (f) => goJob(await api("/api/campaign","POST",Object.fromEntries(f))));
form("#briefForm", async (f) => {
  let b = Object.fromEntries(f);
  b.silent = f.has("silent");
  b.use_footage = f.has("use_footage");
  b.variants = Number(b.variants);
  b.shots = Number(b.shots);
  b.reference = b.reference || null;
  goJob(await api("/api/create", "POST", b));
});
form("#compareForm", async (f) => {
  const candidates = f.getAll("candidate").map((value) => {
    const [job, v] = value.split("/");
    return { job, variant: Number(v) || null };
  });
  if (candidates.length < 2 || candidates.length > 10)
    throw Error("2 ile 10 farklı video seç.");
  goJob(
    await api("/api/compare", "POST", { brief: f.get("brief"), candidates }),
  );
});
form("#uploadForm", async (f) => goJob(await api("/api/analyze", "POST", f)));
form("#settingsForm", async (f) => {
  let data = Object.fromEntries(
    [...f].filter(
      ([k, v]) =>
        (!k.endsWith("TOKEN") && !k.endsWith("SECRET") && k !== "LLM_KEY") || v,
    ),
  );
  await api("/api/settings", "POST", data);
  await settings();
  note("Bağlantılar kaydedildi.");
});
form("#catalogForm", async (f) => {
  let rows = await api("/api/civitai?q=" + encodeURIComponent(f.get("q")));
  $("#catalogResults").innerHTML =
    rows
      .map(
        (r) =>
          "<article><h3>" +
          esc(r.name) +
          "</h3><p>" +
          esc(r.trigger_words.join(", ")) +
          " · " +
          esc(r.base) +
          "</p><p>Ticari izinler · " +
          esc(JSON.stringify(r.commercial_use)) +
          "</p><small>" +
          esc(r.examples[0] || "") +
          '</small><p><button data-lora="' +
          Number(r.version) +
          '">İndir ve etkinleştir</button></p></article>',
      )
      .join("") || "<p>Bu aramada H3 uyumlu LoRA bulunamadı.</p>";
});
async function settings() {
  const [v, h] = await Promise.all([api("/api/settings"), api("/api/models")]);
  const input = (k, s) =>
    "<label>" +
    esc(fields[k]) +
    '<input name="' +
    k +
    '" ' +
    (s
      ? 'type="password" autocomplete="new-password" placeholder="' +
        (v[k] ? "Kayıtlı" : "") +
        '"'
      : 'value="' + esc(v[k]) + '"') +
    "></label>";
  $("#settingsFields").innerHTML =
    Object.keys(fields)
      .filter(
        (k) => !k.endsWith("TOKEN") && !k.endsWith("SECRET") && k !== "LLM_KEY",
      )
      .map((k) => input(k, false))
      .join("") +
    '<label>Brief ile LoRA keşfi<select name="AUTO_LORA"><option value="off">Kapalı</option><option value="on" ' +
    (v.AUTO_LORA === "on" ? "selected" : "") +
    ">Uyumlu ticari LoRA indir ve kullan</option></select></label>";
  $("#secretFields").innerHTML = Object.keys(fields)
    .filter(
      (k) => k.endsWith("TOKEN") || k.endsWith("SECRET") || k === "LLM_KEY",
    )
    .map((k) => input(k, true))
    .join("");
  $("#hardware").innerHTML =
    "<h2>" +
    esc(h.gpu || "Yerel işlemci") +
    "</h2><p>" +
    esc(h.vram_gib ?? "?") +
    " GiB VRAM</p>" +
    h.models
      .map(
        (m) =>
          "<div><b>" +
          esc(m.id) +
          "</b> · " +
          (m.available ? "Hazır" : "Eksik") +
          " · öneri " +
          m.rank +
          " · " +
          (m.history.tests
            ? m.history.tests +
              " deneme · ort. kalite " +
              m.history.quality.toFixed(1) +
              " · geçiş %" +
              Math.round(100 * m.history.pass_rate) +
              " · ort. " +
              Math.round(m.history.seconds) +
              " sn"
            : "başlangıç tahmini") +
          "<p>Başlangıç " +
          m.prior +
          " · geçmiş " +
          (m.history_delta >= 0 ? "+" : "") +
          m.history_delta +
          " · donanım " +
          m.hardware_delta +
          " · profil " +
          m.planning_vram_gib +
          " GiB</p>" +
          (m.missing.length
            ? "<details><summary>Gereken dosyalar ve bileşenler</summary>" +
              esc(m.missing.join(", ")) +
              (m.id !== "standalone"
                ? '<p><button data-model="' +
                  m.id +
                  '">Model dosyalarını kur</button></p>'
                : "") +
              "</details>"
            : "") +
          "</div>",
      )
      .join("");
}
function list(items) {
  return (
    "<ul>" + items.map((i) => "<li>" + esc(i) + "</li>").join("") + "</ul>"
  );
}
function review(v) {
  if (!v) return "";
  if (v.status !== "assessed")
    return (
      "<p>Görsel model değerlendirmesi alınamadı. " + esc(v.reason) + "</p>"
    );
  return (
    "<h3>Görsel yönetmen</h3><p>" +
    esc(v.style) +
    "</p><b>Açılış</b><p>" +
    esc(v.hook) +
    "</p><b>Olası hedef kitle</b><p>" +
    esc(v.audience) +
    "</p><b>Planların işlevi</b>" +
    list(v.beats) +
    "<b>Gözlemler</b>" +
    list(v.evidence) +
    (v.evidence_by_check?.length
      ? "<b>Görsel kabul koşulları</b>" +
        list(
          v.evidence_by_check.map(
            (x) =>
              (({
                met: "Geçti",
                contradicted: "Karşılanmadı",
                unclear: "Belirsiz",
              })[x.result] || x.result) +
              " · " +
              (v.checks?.[x.check_id - 1] || "") +
              " · " +
              x.observation,
          ),
        )
      : "") +
    "<b>İyileştirmeler</b>" +
    list(v.improvements) +
    ((v.severe_visual_defects || []).length
      ? "<b>Önemli görsel sorunlar</b>" + list(v.severe_visual_defects)
      : "")
  );
}
async function jobView(id) {
  const j = await api("/api/jobs/" + id);
  if (location.hash !== "#job/" + id) return;
  const r = j.result || {},
    win = r.winner,
    rank = r.ranking || [],
    v = win?.variant || rank[0]?.variant;
  const clip = win
    ? "winner.mp4"
    : j.assets.includes("input.mp4")
      ? "input.mp4"
      : j.assets.find((p) => p === "v" + v + "/master.mp4");
  const dir =
    j.kind === "reference"
      ? "analysis"
      : j.assets.includes("v" + v + "/check-repair/report.json")
        ? "v" + v + "/check-repair"
        : "v" + v + "/check";
  const get = (p) => fetch(fileUrl(id, p)).then((x) => x.json());
  const report = j.assets.includes(dir + "/report.json")
    ? await get(dir + "/report.json")
    : null;
  const critPath = "v" + v + "/critique.json",
    crit =
      r.critique || (j.assets.includes(critPath) ? await get(critPath) : null);
  const body = $("#jobBody");
  const sources =
    j.kind === "create" && ["complete", "needs_review"].includes(j.status)
      ? rank.filter((x) =>
          ["master.mp4", "story.json"].every((f) =>
            j.assets.includes("v" + x.variant + "/" + f),
          ),
        )
      : [];
  body.innerHTML =
    '<div class="intro"><h1>' +
    esc(label[j.kind] || j.kind) +
    "</h1><p>" +
    esc(label[j.status] || j.status) +
    " · " +
    esc(j.detail) +
    "</p></div>" +
    (j.revision
      ? '<p><a href="#job/' +
        esc(j.revision.job) +
        '">Kaynak kurgu · V' +
        esc(j.revision.variant) +
        "</a> · " +
        esc(j.revision.changes) +
        "</p>"
      : "") +
    (clip
      ? '<div class="split"><div><video id="player" controls playsinline src="' +
        fileUrl(id, clip) +
        '"></video><a href="' +
        fileUrl(id, clip) +
        '" download>MP4 indir ↓</a></div><div>' +
        (report
          ? '<h2>Kare denetimi</h2><span class="chip">' +
            report.frames +
            ' kare</span><span class="chip">' +
            report.duration +
            ' sn</span><span class="chip">Teknik puan ' +
            report.score +
            "</span><p>" +
            Object.entries(report.flag_counts)
              .map(([k, n]) => esc(k) + " · " + n)
              .join(" / ") +
            '</p><div id="frameChecks" class="checks"></div>'
          : "") +
        review(crit) +
        "</div></div>"
      : review(crit)) +
    (rank.length
      ? '<div class="card"><h2>Varyant karşılaştırması</h2>' +
        rank
          .map(
            (x) =>
              "<article><b>V" +
              x.variant +
              " · " +
              esc(x.angle) +
              "</b> · " +
              x.score +
              "/100 · " +
              esc(x.provider) +
              " · " +
              (x.passed ? "Geçti" : "Elenmiş") +
              " · " +
              x.seconds +
              " sn " +
              (j.assets.includes("v" + x.variant + "/master.mp4")
                ? '<a href="' +
                  fileUrl(id, "v" + x.variant + "/master.mp4") +
                  '">İzle</a>'
                : "") +
              "<p>" +
              esc(x.reasons.join(", ")) +
              "</p>" +
              (j.assets.includes("v" + x.variant + "/critique.json")
                ? '<details data-review="' +
                  x.variant +
                  '"><summary>Değerlendirme gerekçesi</summary><div></div></details>'
                : "") +
              "</article>",
          )
          .join("") +
        "</div>"
      : "") +
    (sources.length
      ? '<form class="card" id="revisionForm"><h2>Bu kurguyu revize et</h2><label>Kaynak varyant<select name="variant">' +
        sources
          .map(
            (x) =>
              '<option value="' +
              x.variant +
              '"' +
              (x.variant === v ? " selected" : "") +
              ">V" +
              x.variant +
              " · " +
              esc(x.angle) +
              "</option>",
          )
          .join("") +
        '</select></label><label>Neyi değiştirelim?<textarea name="changes" minlength="3" maxlength="1500" required></textarea></label><button>Revizyon üret</button></form>'
      : "") +
    (r.analysis
      ? '<div class="card"><h2>Kurgu ve metin</h2>' +
        list(
          r.analysis.shots
            .slice(0, 50)
            .map((s) => s.start + "–" + s.end + " sn"),
        ) +
        list(
          r.analysis.frequent_text.map((x) => x[0] + " · " + x[1] + " kare"),
        ) +
        (r.transcript?.segments
          ? list(r.transcript.segments.map((s) => s.start + "s · " + s.text))
          : "") +
        '<button id="useRef">Bu referansla üret</button></div>'
      : "") +
    (win
      ? '<div class="card"><h2>Dağıtım</h2><label>Hedef<select id="channel"><option value="facebook">Facebook Reels</option><option value="meta_asset">Meta Ads · video kitaplığı</option><option value="tiktok">TikTok Direct Post</option></select></label><label>Gönderi metni<input id="postTitle" maxlength="2000"></label><div id="creator"></div><label class="choice"><input type="checkbox" id="consent">Önizlediğim videonun seçili hesaba yüklenmesini onaylıyorum.</label><button id="publish">Videoyu gönder</button><div id="deliveries">' +
        j.deliveries
          .map(
            (d) =>
              "<p>" +
              esc(d.channel) +
              " · " +
              esc(d.status) +
              ' <button data-status="' +
              esc(d.channel) +
              '">Durumu sorgula</button></p>',
          )
          .join("") +
        "</div></div>"
      : "") +
    (rank.length && j.kind === "create"
      ? '<form id="metrics" class="card"><h2>Kampanya sonucu</h2><div class="grid">' +
        ["variant", "impressions", "clicks", "conversions"]
          .map(
            (k, i) =>
              "<label>" +
              ["Varyant", "Gösterim", "Tıklama", "Dönüşüm"][i] +
              '<input name="' +
              k +
              '" type="number" min="' +
              (i < 2 ? 1 : 0) +
              '" required></label>',
          )
          .join("") +
        '</div><div class="bar"><small>Aynı varyantın toplamını günceller.</small><button>Sonucu kaydet</button></div></form>'
      : "") +
    (["needs_review", "interrupted", "failed"].includes(j.status) &&
    ["create","campaign"].includes(j.kind)
      ? '<button id="retry">Yeni deneme başlat</button>'
      : "") +
    "<details><summary>Üretim kayıtları ve kare raporları</summary>" +
    j.assets
      .map(
        (p) =>
          '<a class="file" href="' + fileUrl(id, p) + '">' + esc(p) + "</a>",
      )
      .join("") +
    "</details>";
  for (const panel of body.querySelectorAll("details[data-review]")) {
    panel.ontoggle = async () => {
      if (!panel.open || panel.dataset.loaded) return;
      try {
        const model = await get("v" + panel.dataset.review + "/critique.json");
        panel.querySelector("div").innerHTML =
          (model.status === "assessed"
            ? "<p>Brief uyumu · " + esc(model.brief_fit) + "/100</p>"
            : "") + review(model);
        panel.dataset.loaded = "1";
      } catch (err) {
        note(err.message);
      }
    };
  }
  if ($("#frameChecks")) {
    const box = $("#frameChecks");
    box.innerHTML =
      '<label>Kare <input id="scrub" type="range" min="0" max="' +
      (report.frames - 1) +
      '" value="0"></label><div class="inline"><button id="prevFrame">← Önceki</button><button id="nextFrame">Sonraki →</button></div><p id="frameText"></p><div id="issueList"></div>';
    const base =
      "/api/jobs/" + id + "/frames?report=" + encodeURIComponent(dir);
    let page = [],
      requestNumber = 0;
    async function showFrame(index) {
      const seq = ++requestNumber;
      index = Math.max(0, Math.min(report.frames - 1, index));
      $("#scrub").value = index;
      if (!page.some((x) => x.frame === index)) {
        const data = await api(
          base + "&start=" + Math.floor(index / 100) * 100 + "&limit=100",
        );
        if (seq !== requestNumber) return;
        page = data;
      }
      const x = page.find((x) => x.frame === index);
      if (!x) return;
      $("#player").pause();
      $("#player").currentTime = x.time + 0.0001;
      $("#frameText").textContent =
        "#" +
        x.frame +
        " · " +
        x.time.toFixed(2) +
        "s · " +
        (x.ocr.map((l) => l.text).join(" / ") || "Yazı görülmedi") +
        (x.expected ? " · Beklenen “" + x.expected + "”" : "") +
        (x.glyph_match != null
          ? " · Glif " + Math.round(x.glyph_match * 100) + "%"
          : "") +
        " · " +
        (x.flags.join(", ") || "Kusur işareti yok");
    }
    $("#scrub").oninput = (e) => showFrame(Number(e.target.value));
    $("#prevFrame").onclick = () => showFrame(Number($("#scrub").value) - 1);
    $("#nextFrame").onclick = () => showFrame(Number($("#scrub").value) + 1);
    await showFrame(0);
    const issues = await api(base + "&issues_only=true&limit=20");
    $("#issueList").innerHTML = issues
      .map(
        (x) =>
          '<button data-frame="' +
          x.frame +
          '">#' +
          x.frame +
          " · " +
          esc(x.flags.join(", ")) +
          "</button>",
      )
      .join("");
    $("#issueList").onclick = (e) => {
      if (e.target.dataset.frame) showFrame(Number(e.target.dataset.frame));
    };
  }
  if ($("#revisionForm"))
    form("#revisionForm", async (f) =>
      goJob(
        await api("/api/jobs/" + id + "/revise", "POST", {
          variant: Number(f.get("variant")),
          changes: f.get("changes").trim(),
        }),
      ),
    );
  if ($("#useRef"))
    $("#useRef").onclick = () => {
      sessionStorage.setItem("framecraft-ref", id);
      location.hash = "create";
    };
  if ($("#metrics"))
    form("#metrics", async (f) => {
      await api(
        "/api/jobs/" + id + "/metrics",
        "POST",
        Object.fromEntries([...f].map(([k, v]) => [k, Number(v)])),
      );
      note("Sonuç kaydedildi.");
    });
  if ($("#metrics")) {
    const form = document.createElement("form");
    form.className = "card";
    form.innerHTML =
      '<details><summary>Meta Ads sonuçlarını içe aktar</summary><div class="grid"><label>Reklam ID<input name="ad_id" pattern="[0-9]+" required></label><label>Varyant<input name="variant" type="number" min="1" value="1" required></label><label>Başlangıç<input name="since" type="date" required></label><label>Bitiş<input name="until" type="date" required></label></div><button>Meta’dan ölçümleri al</button></details>';
    $("#metrics").after(form);
    form.onsubmit = async (e) => {
      e.preventDefault();
      try {
        let d = Object.fromEntries(new FormData(form));
        d.variant = Number(d.variant);
        let r = await api("/api/jobs/" + id + "/insights/meta", "POST", d);
        note(r.impressions + " gösterim, " + r.clicks + " tıklama kaydedildi.");
      } catch (err) {
        note(err.message);
      }
    };
  }
  if ($("#retry"))
    $("#retry").onclick = async () => {
      const next = await api("/api/jobs/" + id + "/retry", "POST");
      if (next.id === id) await jobView(id);
      else goJob(next);
    };
  if ($("#channel"))
    $("#channel").onchange = async () => {
      const box = $("#creator");
      box.innerHTML = "";
      $("#consent").checked = false;
      if ($("#channel").value !== "tiktok") return;
      try {
        let c = await api("/api/tiktok/creator");
        box.innerHTML =
          "<p>TikTok · " +
          esc(c.creator_nickname) +
          '</p><label>Görünürlük<select id="privacy"><option value="">Seç</option>' +
          c.privacy_level_options
            .map((o) => "<option>" + esc(o) + "</option>")
            .join("") +
          "</select></label>" +
          [
            ["comments", "Yorum", c.comment_disabled],
            ["duet", "Düet", c.duet_disabled],
            ["stitch", "Ekleme", c.stitch_disabled],
            ["own_brand", "Kendi markam", false],
            ["branded", "Ücretli marka işbirliği", false],
          ]
            .map(
              ([k, t, off]) =>
                '<label class="choice"><input type="checkbox" id="' +
                k +
                '" ' +
                (off ? "disabled" : "") +
                ">" +
                t +
                "</label>",
            )
            .join("") +
          '<p><a href="https://www.tiktok.com/legal/page/global/music-usage-confirmation/en" target="_blank">Music Usage Confirmation</a> · <a href="https://www.tiktok.com/legal/page/global/bc-policy/en" target="_blank">Branded Content Policy</a></p><label class="choice"><input type="checkbox" id="terms">Müzik kullanım onayını ve ilgili marka içerik koşullarını kabul ediyorum.</label>';
      } catch (e) {
        note(e.message);
      }
    };
  if ($("#publish"))
    $("#publish").onclick = async () => {
      try {
        let ch = $("#channel").value;
        if (ch === "tiktok" && !$("#terms")?.checked)
          throw Error("TikTok koşullarını onayla.");
        const opt = {
          title: $("#postTitle").value,
          consent: $("#consent").checked,
          privacy: $("#privacy")?.value,
        };
        for (const k of ["comments", "duet", "stitch", "own_brand", "branded"])
          opt[k] = !!$("#" + k)?.checked;
        await api("/api/jobs/" + id + "/publish/" + ch, "POST", opt);
        await jobView(id);
        note("Yükleme gönderildi. Durumunu sorgulayabilirsin.");
      } catch (e) {
        note(e.message);
      }
    };
  if (["queued", "running"].includes(j.status)) timer = setTimeout(route, 3000);
}
let timer;
async function route() {
  clearTimeout(timer);
  const [section = "create", id] = location.hash.slice(1).split("/"),
    page = ["create", "reference", "projects", "jobs", "job", "settings"].includes(section)
      ? section
      : "create";
  document
    .querySelectorAll("main>section")
    .forEach((el) => (el.hidden = el.id !== page));
  document
    .querySelectorAll("nav a")
    .forEach((a) =>
      a.classList.toggle(
        "active",
        a.hash === "#" + (page === "job" ? "jobs" : page),
      ),
    );
  try {
    if (page === "projects") {
      const projects = await api("/api/projects");
      $("#projectList").innerHTML = projects.map(project => {
        const videos = project.editions.map((edition, i) => {
          const url = "/api/projects/" + encodeURIComponent(project.name) + "/" + edition.number + "/video";
          const label = edition.stage === "preview" ? " · Ön izleme" : "";
          return '<details' + (i === 0 ? " open" : "") + '><summary>UGC ' +
            String(edition.number).padStart(3, "0") + label +
            '</summary><video controls playsinline preload="none" style="max-width:100%;max-height:65vh" src="' +
            esc(url) + '"></video></details>';
        }).join("");
        return '<article class="card"><h2>' + esc(project.name) + '</h2>' +
          (videos || "<p>Henüz video yok.</p>") + '</article>';
      }).join("") || "<p>Henüz proje yok.</p>";
    }
    if (page === "settings") await settings();
    if (page === "jobs" || page === "create") {
      const rows = await api("/api/jobs");
      if (page === "jobs") {
        const options = await api("/api/candidates");
        $("#candidateList").innerHTML = options
          .map(
            (c) =>
              '<label class="choice"><input type="checkbox" name="candidate" value="' +
              esc(c.id) +
              "/" +
              (c.variant || 0) +
              '">' +
              esc(c.provider || "Referans") +
              " " +
              (c.variant ? "V" + c.variant : "") +
              " · " +
              new Date(c.updated * 1000).toLocaleString("tr") +
              "</label>",
          )
          .join("");
      }
      if (page === "jobs")
        $("#jobList").innerHTML =
          rows
            .map(
              (r) =>
                '<a class="jobrow" href="#job/' +
                r.id +
                '"><b>' +
                esc(label[r.kind]) +
                " · " +
                esc(label[r.status]) +
                "</b><small>" +
                esc(r.detail) +
                " · " +
                new Date(r.updated * 1000).toLocaleString("tr") +
                "</small></a>",
            )
            .join("") || "<p>Henüz kayıt yok.</p>";
      else {
        const ref = $("#refs").value;
        $("#refs").innerHTML =
          '<option value="">Referans kullanma</option>' +
          rows
            .filter((r) => r.kind === "reference" && r.status === "complete")
            .map(
              (r) =>
                '<option value="' +
                r.id +
                '">' +
                esc(r.detail) +
                " · " +
                new Date(r.updated * 1000).toLocaleTimeString("tr") +
                "</option>",
            )
            .join("");
        $("#refs").value =
          sessionStorage.getItem("framecraft-ref") || ref || "";
        sessionStorage.removeItem("framecraft-ref");
      }
    }
    if (page === "job") await jobView(id);
  } catch (e) {
    note(e.message);
  }
}
document.addEventListener("click", async (e) => {
  let b = e.target.closest("button");
  if (!b) return;
  try {
    if (b.dataset.seek && $("#player")) {
      $("#player").currentTime = Number(b.dataset.seek);
      $("#player").pause();
    }
    if (b.dataset.connect) {
      const r = await api("/api/connect/" + b.dataset.connect, "POST");
      location.assign(r.authorize);
    }
    if (b.id === "getPages") {
      const rows = await api("/api/meta/pages");
      $("#pages").innerHTML =
        rows
          .map(
            (p) =>
              '<button data-page="' +
              esc(p.id) +
              '">' +
              esc(p.name) +
              "</button>",
          )
          .join("") || "Sayfa bulunamadı";
    }
    if (b.dataset.page) {
      await api("/api/meta/pages/" + b.dataset.page, "POST");
      await settings();
      note("Facebook sayfası bağlandı.");
    }
    if (b.dataset.model)
      goJob(await api("/api/install/model/" + b.dataset.model, "POST"));
    if (b.dataset.lora)
      goJob(await api("/api/install/lora/" + b.dataset.lora, "POST"));
    if (b.dataset.status) {
      let r = await api(
        "/api/jobs/" +
          location.hash.split("/")[1] +
          "/delivery/" +
          b.dataset.status,
        "POST",
      );
      note(r.status);
    }
  } catch (err) {
    note(err.message);
  }
});
window.addEventListener("hashchange", () => {
  note("");
  route();
});
route();
