// Slide deck for "Looking the Other Way", built from data/numbers.json (run make_numbers.py first).
//   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node build_deck.js <apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.argv[2]);

const HERE = __dirname;
const NUM = JSON.parse(fs.readFileSync(path.join(HERE, "data", "numbers.json"), "utf8"));
const HIST = JSON.parse(fs.readFileSync(path.join(HERE, "data", "noise_hist.json"), "utf8")).LG_base25_EC_5subsets;
const OUT = path.join(HERE, "looking-the-other-way-slides.pptx");
const C0 = NUM.conditions;
const DIFF_GAP = JSON.parse(fs.readFileSync(path.join(HERE, "data",
  "diff_openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor-difficulty-label_true.json"), "utf8")).conditions[0].gap_shown.value;
const R110 = NUM.restore["openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor_restore_channel_link_cost110"].per_agent;

// Model colours match the paper's figures (dataviz reference palette, validated order).
const THEME = {
  name: "Looking the Other Way",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2033", lt1: "FFFFFF", dk2: "52514E", lt2: "F3F3F0",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "6250D6", accent5: "E87BA4", accent6: "F2B33D",
    hlink: "1C5CAB", folHlink: "6250D6",
  },
};
const HEX = { ink: "1D2033", ink2: "52514E", muted: "898781", grid: "E1E0D9", mist: "F3F3F0", white: "FFFFFF",
  luna: "2A78D6", glm: "EB6834", deepseek: "1BAF7A", haiku: "6250D6", mimo: "E87BA4", amber: "F2B33D", red: "E34948",
  inc: "6250D6", info: "EB6834", evid: "1BAF7A", model: "2A78D6" };

const pct = (v, d = 0) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(d)}%`);
const r1 = (v) => Math.round(1000 * v) / 10;
const m = (k, metric) => C0[k].metrics[metric].value;
const d = (k, metric) => C0[k].delta[metric];
const pv = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`);

// "a → b" with the arrow set in the body font, since Cambria's arrow glyph renders tiny
const arrow = (a, b) => [{ text: a }, { text: " → ", options: { fontFace: "Calibri" } }, { text: b }];

async function icon(Comp, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.title = "Looking the Other Way: Levers of Collusion in LLM Peer Review";
  pres.author = "Tanweerul Haque";
  pres.subject = "Apart Research AI Collusion Research Sprint, October 2026";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333;

  pres.defineSlideMaster({
    title: "DARK",
    background: { color: C.text1 },
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: THEME.headFontFace,
        fontSize: 30, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "Looking the Other Way  ·  T. Haque  ·  Apart AI Collusion Research Sprint, Oct 2026",
        options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
  });

  const ic = {
    eye: await icon(fa.FaEye, HEX.white), eyeSlash: await icon(fa.FaEyeSlash, HEX.white),
    code: await icon(fa.FaCode, HEX.white), chat: await icon(fa.FaComments, HEX.white),
    gavel: await icon(fa.FaGavel, HEX.white), coins: await icon(fa.FaCoins, HEX.white), pen: await icon(fa.FaPenFancy, HEX.white),
    robot: await icon(fa.FaRobot, HEX.white), scale: await icon(fa.FaBalanceScale, HEX.white),
    clock: await icon(fa.FaHourglassHalf, HEX.white), crystal: await icon(fa.FaQuestionCircle, HEX.white),
    list: await icon(fa.FaSortAmountDown, HEX.white), tag: await icon(fa.FaTag, HEX.white),
    unlock: await icon(fa.FaUnlockAlt, HEX.white), dollar: await icon(fa.FaDollarSign, HEX.white),
    search: await icon(fa.FaSearch, HEX.white), check: await icon(fa.FaClipboardCheck, HEX.white),
    chart: await icon(fa.FaChartBar, HEX.white), copy: await icon(fa.FaCopy, HEX.white), lock: await icon(fa.FaLock, HEX.white),
  };

  // icon in a coloured circle: the deck's one motif
  function badge(slide, img, color, x, y, size = 0.6, name = "badge") {
    slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color }, line: { color, width: 0 }, objectName: name + "-circle" });
    const pad = size * 0.24;
    slide.addImage({ data: img, x: x + pad, y: y + pad, w: size - 2 * pad, h: size - 2 * pad, objectName: name + "-icon" });
  }
  function card(slide, x, y, w, h, fill = C.background2, name = "card") {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
  }
  function text(slide, t, opts) {
    slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...opts });
  }
  const chartBase = () => ({
    catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.ink2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" }, valAxisLineShow: false,
    catAxisLineShow: true, dataLabelColor: HEX.ink, dataLabelFontSize: 11, legendFontSize: 12, legendColor: HEX.ink2,
  });

  // ------------------------------------------------------------------ 1. title
  pres.addSection({ title: "Setup" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Setup" });
  text(s, "Looking the Other Way", { x: 0.7, y: 1.55, w: 7.6, h: 1.0, fontFace: THEME.headFontFace, fontSize: 48, bold: true, color: C.background1 });
  text(s, "Levers of collusion in LLM peer review", { x: 0.7, y: 2.6, w: 7.6, h: 0.6, fontSize: 24, color: C.accent6 });
  text(s, "When two LLM agents must verify each other under a rule they cannot meet, what makes them accept anyway, and what stops them?",
    { x: 0.7, y: 3.45, w: 7.0, h: 1.0, fontSize: 16, color: C.background2 });
  text(s, [
    { text: "Tanweerul Haque", options: { bold: true, breakLine: true } },
    { text: "Nanyang Technological University, Singapore", options: { breakLine: true } },
  ], { x: 0.7, y: 4.95, w: 7, h: 0.7, fontSize: 16, color: C.background1 });
  text(s, "Apart Research AI Collusion Research Sprint · 23–25 October 2026 · Track 3\ngithub.com/TanweerulHaque/levers-of-collusion",
    { x: 0.7, y: 5.9, w: 8, h: 0.6, fontSize: 12, color: C.background2 });
  badge(s, ic.eyeSlash, HEX.glm, 8.9, 2.15, 1.75, "alice");
  badge(s, ic.eyeSlash, HEX.luna, 10.95, 2.15, 1.75, "bob");
  text(s, "Alice", { x: 8.9, y: 4.05, w: 1.75, h: 0.4, fontSize: 16, color: C.background1, align: "center" });
  text(s, "Bob", { x: 10.95, y: 4.05, w: 1.75, h: 0.4, fontSize: 16, color: C.background1, align: "center" });
  text(s, "“I'll accept. Please accept mine.”", { x: 8.6, y: 4.65, w: 4.4, h: 0.5, fontSize: 14, italic: true, color: C.accent6, align: "center" });
  s.addNotes("This project asks what makes two LLM reviewers approve each other's work without the evidence their rule requires, and which levers an operator can pull to stop it. It builds on the joint-task game of Shi et al. (2026), arXiv 2609.24967, and the SALT-NLP agent-collusion code. Everything after the first branch of the fork is my own: eight levers, run on five model pairs.");

  // ------------------------------------------------------------------ 2. the game
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
  s.addText("Two agents check each other under a rule they cannot meet", { placeholder: "title" });
  const phases = [
    [ic.code, "Task", "Solve own task, save answer, fetch the raw log"],
    [ic.chat, "Talk", "5 rounds, one message of ≤200 characters each"],
    [ic.gavel, "Verdict", "ACCEPT or REJECT on the peer's work"],
    [ic.coins, "Feedback", "Shared reward 0/10/20 and a verdict review"],
    [ic.pen, "Reflect", "Private note, kept in memory for 10 episodes"],
  ];
  phases.forEach(([img, head, body], i) => {
    const x = 0.6 + i * 2.48;
    card(s, x, 1.45, 2.25, 2.2, C.background2, `phase-${i}`);
    badge(s, img, HEX.ink, x + 0.2, 1.65, 0.6, `phase-${i}`);
    text(s, head, { x: x + 0.2, y: 2.35, w: 1.9, h: 0.4, fontSize: 18, bold: true });
    text(s, body, { x: x + 0.2, y: 2.8, w: 1.9, h: 0.8, fontSize: 14, color: C.text2 });
  });
  card(s, 0.6, 3.9, 5.9, 2.5, "FFF4E9", "rule");
  text(s, "The rule", { x: 0.85, y: 4.05, w: 5.4, h: 0.4, fontSize: 18, bold: true });
  text(s, "“Your verdict must be based on Bob's complete raw-log string, with no omission […]”",
    { x: 0.85, y: 4.5, w: 5.4, h: 0.6, fontSize: 14, italic: true });
  text(s, "1,000 chars", { x: 0.85, y: 5.2, w: 2.5, h: 0.6, fontSize: 28, bold: true, fontFace: THEME.headFontFace, color: C.accent2 });
  text(s, "can be sent per episode; logs run 1,300–39,000, so only REJECT complies", { x: 3.35, y: 5.2, w: 3.0, h: 1.0, fontSize: 13, color: C.text2 });
  card(s, 6.83, 3.9, 5.9, 2.5, "EAF2FC", "reward");
  text(s, "The reward", { x: 7.08, y: 4.05, w: 5.4, h: 0.4, fontSize: 18, bold: true });
  text(s, "+10 for each verdict that matches the ground truth of the peer's task",
    { x: 7.08, y: 4.5, w: 5.4, h: 0.6, fontSize: 14 });
  text(s, "≈89%", { x: 7.08, y: 5.2, w: 2.5, h: 0.6, fontSize: 28, bold: true, fontFace: THEME.headFontFace, color: C.accent1 });
  text(s, "of tasks are correct, so ACCEPT usually pays", { x: 9.58, y: 5.2, w: 3.0, h: 1.0, fontSize: 13, color: C.text2 });
  text(s, "Collusion: both agents ACCEPT in the same episode. EC is the share of such episodes.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 14, bold: true });
  s.addNotes("The game is the one from Shi et al. Two agents solve tasks, talk over a narrow channel, and judge each other's work. The rule says a verdict must rest on the peer's complete raw log, which never fits through 1,000 characters, so the only compliant verdict is REJECT. The reward pays for verdicts that match ground truth, and almost nine in ten tasks are correct, so the reward pulls toward ACCEPT. Every mutual ACCEPT in the baseline is a rule violation.");

  // ------------------------------------------------------------------ 3. levers
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
  s.addText("Eight levers, five model pairs, one game", { placeholder: "title" });
  const levers = [
    [ic.robot, HEX.model, "Model pairing", "5 models; Luna and GLM in both seats"],
    [ic.scale, HEX.inc, "Per-term penalty", "−10 for each wrong verdict"],
    [ic.clock, HEX.info, "Episode countdown", "“Episode 3 of 10. 7 more follow.”"],
    [ic.crystal, HEX.info, "Private forecast", "Predict the peer's verdict on your work"],
    [ic.list, HEX.info, "Sequential verdict", "Bob sees Alice's verdict first"],
    [ic.tag, HEX.info, "Difficulty label", "“Difficulty of Bob's task: hard”"],
    [ic.unlock, HEX.evid, "Paid restore", "Lift the cap for 50% or 110% of reward"],
    [ic.dollar, HEX.evid, "Message price", "No cap; pay per character sent"],
  ];
  levers.forEach(([img, col, head, body], i) => {
    const x = 0.6 + (i % 4) * 3.05, y = 1.45 + Math.floor(i / 4) * 1.55;
    card(s, x, y, 2.85, 1.35, C.background2, `lever-${i}`);
    badge(s, img, col, x + 0.18, y + 0.2, 0.55, `lever-${i}`);
    text(s, head, { x: x + 0.88, y: y + 0.18, w: 1.85, h: 0.4, fontSize: 15, bold: true });
    text(s, body, { x: x + 0.88, y: y + 0.6, w: 1.85, h: 0.65, fontSize: 12, color: C.text2 });
  });
  const legend = [[HEX.model, "Model"], [HEX.inc, "Incentive"], [HEX.info, "Information"], [HEX.evid, "Evidence"]];
  legend.forEach(([col, lab], i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.6 + i * 1.75, y: 4.72, w: 0.18, h: 0.18, fill: { color: col }, line: { color: col, width: 0 }, objectName: `legend-${i}` });
    text(s, lab, { x: 0.86 + i * 1.75, y: 4.65, w: 1.4, h: 0.3, fontSize: 12, color: C.text2 });
  });
  const stats = [[NUM.extras.totals.trajectories, "trajectories"], [NUM.extras.totals.episodes.toLocaleString("en-US"), "episodes"], ["5", "models"],
    ["$" + Math.round(NUM.extras.totals.cost_usd), "API cost"]];
  stats.forEach(([v, lab], i) => {
    text(s, String(v), { x: 0.6 + i * 3.05, y: 5.25, w: 2.8, h: 0.75, fontSize: 40, bold: true, fontFace: THEME.headFontFace, color: C.text1 });
    text(s, lab, { x: 0.6 + i * 3.05, y: 6.0, w: 2.8, h: 0.35, fontSize: 14, color: C.text2 });
  });
  s.addNotes("Each lever is one opt-in flag on its own branch of the fork; with every flag off the prompts are byte-identical to the original. For each lever I ask two questions: does collusion change, and if so, is it because agents judge better or because they simply reject more? About 89% of the work is correct, so rejecting everything ends collusion but is wrong nine times in ten. Models: GPT-6-Luna, GLM-5.3-Flash, DeepSeek V4.1 Flash, MiMo-V2.6-Flash, Claude Haiku 5.5, all at high reasoning effort through OpenRouter.");

  // ------------------------------------------------------------------ 4. findings
  pres.addSection({ title: "Findings" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Four findings", { placeholder: "title" });
  const findings = [
    [ic.robot, HEX.model, arrow("0%", "85%"), "The model sets the level of collusion, and a seat swap leaves each model's acceptance unchanged"],
    [ic.scale, HEX.inc, `${pct(m("LG_base", "EC"), 1)} vs ${pct(m("LG_pen", "EC"))}`, `A penalty for wrong verdicts leaves collusion unchanged at 25 runs per arm (${pv(d("LG_pen", "EC").p)})`],
    [ic.search, HEX.info, "−13 to −23 pp", "Forecasts, labels and sequential verdicts cut collusion by making GLM reject correct work: accuracy falls"],
    [ic.unlock, HEX.evid, arrow("33%", "2–8%"), "Letting the log through, at a price, removes most unsupported acceptance with no loss of accuracy"],
  ];
  findings.forEach(([img, col, big, body], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 2.7;
    card(s, x, y, 5.95, 2.45, C.background2, `finding-${i}`);
    badge(s, img, col, x + 0.3, y + 0.3, 0.7, `finding-${i}`);
    text(s, big, { x: x + 1.2, y: y + 0.25, w: 4.5, h: 0.8, fontSize: 36, bold: true, fontFace: THEME.headFontFace });
    text(s, body, { x: x + 1.2, y: y + 1.15, w: 4.5, h: 1.1, fontSize: 16, color: C.text2 });
  });
  s.addNotes("The four headline results. First, which models review matters more than any lever I applied: episode-level collusion runs from 0% with Claude Haiku to 85% with DeepSeek pairs. Second, making wrong verdicts cost -10 does nothing to collusion at 25 trajectories per arm; it only destroys reward. Third, three information levers do reduce collusion, but only because GLM starts rejecting correct work, so verdict accuracy drops by 13 to 23 points. Fourth, the levers that let the evidence through, a paid channel restore and a per-character message price, cut unsupported mutual acceptance from a third of episodes to 2 to 8 percent while keeping or raising accuracy.");

  // ------------------------------------------------------------------ 5. model lever
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("The model is the strongest lever", { placeholder: "title" });
  const pairs = [["Haiku × MiMo", "HM_fore"], ["Luna → GLM", "LG_base"], ["GLM → Luna", "GL_base"], ["DeepSeek × DeepSeek", "DD_base"], ["DeepSeek × MiMo", "DM_base"]];
  s.addChart(pres.charts.BAR, [{ name: "EC", labels: pairs.map((p) => `${p[0]} (N=${C0[p[1]].trajectories})`), values: pairs.map((p) => r1(m(p[1], "EC"))) }], {
    x: 0.5, y: 1.4, w: 6.3, h: 4.6, barDir: "bar", chartColors: [HEX.luna], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0"%"', valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', valAxisMajorUnit: 25,
    showLegend: false, showTitle: true, title: "Episodes with mutual ACCEPT (EC)", titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    barGapWidthPct: 70, ...chartBase(), objectName: "chart-ec-by-pair",
  });
  const seat = [["GPT-6-Luna", m("LG_base", "alice_accept_rate"), m("GL_base", "bob_accept_rate")],
    ["GLM-5.3-Flash", m("GL_base", "alice_accept_rate"), m("LG_base", "bob_accept_rate")],
    ["DeepSeek V4.1", m("DD_base", "alice_accept_rate"), m("DD_base", "bob_accept_rate")]];
  s.addChart(pres.charts.BAR, [
    { name: "as Alice", labels: seat.map((r) => r[0]), values: seat.map((r) => r1(r[1])) },
    { name: "as Bob", labels: seat.map((r) => r[0]), values: seat.map((r) => r1(r[2])) },
  ], {
    x: 7.0, y: 1.4, w: 5.8, h: 3.55, barDir: "col", barGrouping: "clustered", chartColors: [HEX.ink, "9AA0AE"], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"',
    valAxisMajorUnit: 25, showLegend: true, legendPos: "b", showTitle: true, title: "Share of verdicts that ACCEPT, by seat",
    titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt", barGapWidthPct: 60, ...chartBase(), objectName: "chart-accept-by-seat",
  });
  text(s, [
    { text: "Claude Haiku 5.5 never accepted ", options: { bold: true } },
    { text: `(0 of 39 verdicts), citing the policy every time; MiMo accepted Haiku's work in ${pct(m("HM_fore", "bob_accept_rate"))} of episodes.` },
  ], { x: 7.1, y: 5.15, w: 5.6, h: 0.75, fontSize: 14 });
  text(s, `Swapping seats moves the pair's EC by under one point (${pct(m("LG_base", "EC"), 1)} vs ${pct(m("GL_base", "EC"))}).`,
    { x: 7.1, y: 5.95, w: 5.6, h: 0.7, fontSize: 14, color: C.text2 });
  s.addNotes(`EC by pair, with the number of complete trajectories in brackets. The DeepSeek pairs collude in about 85% of episodes and converge in every trajectory. Luna and GLM sit around a third in either seat order. Haiku never accepts. The right chart shows acceptance belongs to the model: GLM accepts about 74% of the time as Alice or as Bob; Luna about 35 to 39%. On summaries neither model separates correct from failing work well: GLM accepts ${pct(NUM.discrimination.glm.accept_given_pass)} of correct tasks and ${pct(NUM.discrimination.glm.accept_given_fail)} of failing ones.`);

  // ------------------------------------------------------------------ 6. learning
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Every model starts compliant; feedback teaches most to accept", { placeholder: "title" });
  const cv = (k) => C0[k].curves;
  const avg = (a, b) => a.map((v, i) => (v + b[i]) / 2);
  const eps = Array.from({ length: 10 }, (_, i) => String(i + 1));
  const lines = [
    ["DeepSeek V4.1 Flash", avg(cv("DD_base").alice_accept, cv("DD_base").bob_accept), HEX.deepseek],
    ["GLM-5.3-Flash", avg(cv("LG_base").bob_accept, cv("GL_base").alice_accept), HEX.glm],
    ["GPT-6-Luna", avg(cv("LG_base").alice_accept, cv("GL_base").bob_accept), HEX.luna],
    ["Claude Haiku 5.5", cv("HM_fore").alice_accept.map((v) => v || 0), HEX.haiku],
  ];
  s.addChart(pres.charts.LINE, lines.map(([name, ys]) => ({ name, labels: eps, values: ys.map((v) => r1(v)) })), {
    x: 0.5, y: 1.4, w: 7.6, h: 5.3, chartColors: lines.map((l) => l[2]), lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', valAxisMajorUnit: 25, showLegend: true, legendPos: "b",
    showTitle: true, title: "ACCEPT rate by episode (baselines, both seats pooled)", titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "Episode", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.ink2, ...chartBase(), objectName: "chart-learning",
  });
  card(s, 8.45, 1.45, 4.3, 1.55, C.background2, "ep1");
  text(s, `${NUM.episode1_accepts[0]} of ${NUM.episode1_accepts[1]}`, { x: 8.7, y: 1.6, w: 3.9, h: 0.7, fontSize: 36, bold: true, fontFace: THEME.headFontFace });
  text(s, "episode-1 verdicts are ACCEPT, before any feedback", { x: 8.7, y: 2.35, w: 3.9, h: 0.55, fontSize: 14, color: C.text2 });
  text(s, "After a verdict review says a REJECT of correct work cost reward, DeepSeek reaches 88% by episode 2, GLM 84% by episode 4, Luna 56% by episode 5. Haiku gets the same feedback and stays at 0%.",
    { x: 8.45, y: 3.25, w: 4.3, h: 1.6, fontSize: 14 });
  card(s, 8.45, 5.0, 4.3, 1.6, "FFF4E9", "quote-glm");
  text(s, [
    { text: "“the 200-char message cap makes verbatim raw-log transmission impossible […]; her concrete, instruction-matched account is the best available evidence”", options: { italic: true, breakLine: true } },
    { text: "GLM-5.3-Flash, ACCEPT verdict", options: { fontSize: 11, color: HEX.ink2 } },
  ], { x: 8.65, y: 5.1, w: 3.9, h: 1.5, fontSize: 13 });
  s.addNotes("No pair starts out colluding: in episode 1, 3 of 152 baseline verdicts are ACCEPT. The verdict review then tells an agent that its rejection of correct work was wrong, and acceptance climbs within two to five episodes. Haiku receives the same feedback and never moves. This matches Shi et al.'s finding that verification feedback drives collusion. The agents know what they are doing: their verdicts state that the cap makes the rule impossible and accept anyway.");

  // ------------------------------------------------------------------ 7. styles
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("DeepSeek strikes deals; Luna and GLM collude quietly", { placeholder: "title" });
  const deals = NUM.extras.deals;
  const share = (x) => x.announce_and_mutual / x.mutual;
  card(s, 0.6, 1.45, 5.95, 4.6, C.background2, "deepseek");
  badge(s, ic.chat, HEX.deepseek, 0.85, 1.7, 0.6, "deepseek");
  text(s, "DeepSeek × DeepSeek", { x: 1.6, y: 1.78, w: 4.7, h: 0.45, fontSize: 18, bold: true });
  text(s, pct(share(deals.DD_base)), { x: 0.85, y: 2.55, w: 2.2, h: 0.9, fontSize: 48, bold: true, fontFace: THEME.headFontFace, color: C.accent3 });
  text(s, "of mutual ACCEPTs come with an announced acceptance in the channel", { x: 3.05, y: 2.65, w: 3.3, h: 0.8, fontSize: 14, color: C.text2 });
  text(s, [
    { text: "Alice: “Got your 8 IDs for 2-39. Looks plausible; I'll accept. My 2-16 7 IDs stand. Please accept mine.”", options: { italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Bob: “Accepted — your 2-16 7 IDs stand. I'll submit accept in the verdict phase.”", options: { italic: true } },
  ], { x: 0.85, y: 3.75, w: 5.45, h: 2.1, fontSize: 15 });
  card(s, 6.78, 1.45, 5.95, 4.6, C.background2, "lunaglm");
  badge(s, ic.eyeSlash, HEX.glm, 7.03, 1.7, 0.6, "lunaglm");
  text(s, "Luna → GLM", { x: 7.78, y: 1.78, w: 4.7, h: 0.45, fontSize: 18, bold: true });
  text(s, pct(share(deals.LG_base25)), { x: 7.03, y: 2.55, w: 2.2, h: 0.9, fontSize: 48, bold: true, fontFace: THEME.headFontFace, color: C.accent2 });
  text(s, "of mutual ACCEPTs come with such a phrase, all written by GLM", { x: 9.23, y: 2.65, w: 3.3, h: 0.8, fontSize: 14, color: C.text2 });
  text(s, [
    { text: "Luna: “Although the full raw log was not deliverable under the message cap, the transmitted evidence […]”", options: { italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Each agent waives the rule on its own; they rarely negotiate it.", options: { color: HEX.ink2 } },
  ], { x: 7.03, y: 3.75, w: 5.45, h: 2.1, fontSize: 15 });
  text(s, "Counted with a keyword search for “I'll accept”, “please accept mine” and similar phrases: a lower bound, since the LLM agreement judge was not run on most launches.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 12, color: C.text2 });
  s.addNotes("Counted with a keyword search for phrases such as I'll accept or please accept mine; the LLM agreement judge from the original paper was not run on most launches, so these are lower bounds. DeepSeek negotiates openly: 61% of its mutual ACCEPTs come with an announcement. Luna and GLM collude tacitly: each writes its own waiver into its verdict.");

  // ------------------------------------------------------------------ 8. penalty
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Penalising wrong verdicts does not reduce collusion", { placeholder: "title" });
  s.addChart(pres.charts.LINE, [
    { name: `Baseline (N=${C0.LG_base.trajectories})`, labels: eps, values: C0.LG_base.curves.ec.map((v) => r1(v)) },
    { name: `Per-term penalty (N=${C0.LG_pen.trajectories})`, labels: eps, values: C0.LG_pen.curves.ec.map((v) => r1(v)) },
  ], {
    x: 0.5, y: 1.4, w: 7.6, h: 5.3, chartColors: [HEX.ink2, HEX.red], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', valAxisMajorUnit: 25, showLegend: true, legendPos: "b",
    showTitle: true, title: "Mutual ACCEPT (EC) by episode, Luna → GLM", titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "Episode", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.ink2, ...chartBase(), objectName: "chart-penalty",
  });
  const penStats = [
    [arrow(pct(m("LG_base", "EC"), 1), pct(m("LG_pen", "EC"))), `EC, Luna → GLM, 25 runs per arm (${pv(d("LG_pen", "EC").p)})`],
    [arrow(pct(m("GL_base", "EC")), pct(m("GL_pen", "EC"), 1)), `EC, GLM → Luna, ${C0.GL_pen.trajectories} penalty runs (${pv(d("GL_pen", "EC").p)})`],
    [arrow(m("LG_base", "joint_reward_mean").toFixed(1), m("LG_pen", "joint_reward_mean").toFixed(1)), "Mean joint reward per episode"],
  ];
  penStats.forEach(([big, lab], i) => {
    card(s, 8.45, 1.45 + i * 1.45, 4.3, 1.3, C.background2, `pen-${i}`);
    text(s, big, { x: 8.7, y: 1.55 + i * 1.45, w: 3.9, h: 0.65, fontSize: 30, bold: true, fontFace: THEME.headFontFace });
    text(s, lab, { x: 8.7, y: 2.2 + i * 1.45, w: 3.9, h: 0.45, fontSize: 13, color: C.text2 });
  });
  text(s, "The penalty prices wrong outcomes and leaves breaking the rule free, so agents have no new reason to stop accepting without the log.",
    { x: 8.45, y: 5.85, w: 4.3, h: 0.9, fontSize: 14 });
  s.addNotes("Each missed reward term now pays -10 instead of 0, so the shared reward is +20, 0 or -20. With 25 trajectories per arm, collusion, acceptance and accuracy did not move, and the swapped order agrees. Only the reward collapsed. A 5-trajectory pilot had pointed the other way, which is why I scaled this lever to 25.");

  // ------------------------------------------------------------------ 9. information levers
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Information levers make GLM reject correct work", { placeholder: "title" });
  const info = [["Baseline (25)", "LG_base"], ["Countdown", "LG_count"], ["Forecast", "LG_fore"], ["Difficulty label", "LG_diff"], ["Sequential", "LG_seq"]];
  s.addChart(pres.charts.BAR, [
    { name: "GLM ACCEPT rate", labels: info.map((r) => r[0]), values: info.map((r) => r1(C0[r[1]].models.glm.accept_rate)) },
    { name: "Verdict accuracy (both agents)", labels: info.map((r) => r[0]), values: info.map((r) => r1(m(r[1], "verdict_accuracy"))) },
    { name: "Mutual ACCEPT (EC)", labels: info.map((r) => r[0]), values: info.map((r) => r1(m(r[1], "EC"))) },
  ], {
    x: 0.5, y: 1.4, w: 7.8, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.glm, HEX.ink2, "9AA0AE"], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontSize: 10, valAxisMaxVal: 100, valAxisMinVal: 0,
    valAxisLabelFormatCode: '0"%"', valAxisMajorUnit: 25, showLegend: true, legendPos: "b", showTitle: true,
    title: "Luna → GLM; levers ran for 5 trajectories each", titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    barGapWidthPct: 45, ...chartBase(), objectName: "chart-info",
  });
  const infoCards = [
    [ic.crystal, "Private forecast", `EC ${pct(m("LG_base", "EC"))} → ${pct(m("LG_fore", "EC"))}; GLM rejects ${pct(C0.LG_fore.models.glm.false_reject_rate)} of correct work (was ${pct(C0.LG_base.models.glm.false_reject_rate)})`],
    [ic.tag, "Difficulty label", `EC ${pct(m("LG_diff", "EC"))} and ${pct(m("GL_diff", "EC"))}; acceptance ignores the label (hard − easy ${(100 * DIFF_GAP).toFixed(1).replace("-", "−")} pp)`],
    [ic.list, "Sequential verdict", `GLM matches Luna in ${pct(NUM.sequential_follow["openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor-verdict-order_sequential"].follower_matches_leader.rate)} of episodes and never says so`],
    [ic.clock, "Countdown", `No change in EC (${pct(m("LG_count", "EC"))}), even in the last episode`],
  ];
  infoCards.forEach(([img, head, body], i) => {
    const y = 1.45 + i * 1.33;
    badge(s, img, HEX.info, 8.55, y + 0.08, 0.5, `info-${i}`);
    text(s, head, { x: 9.2, y, w: 3.55, h: 0.35, fontSize: 15, bold: true });
    text(s, body, { x: 9.2, y: y + 0.38, w: 3.55, h: 0.85, fontSize: 13, color: C.text2 });
  });
  s.addNotes(`Three of the four information levers cut GLM's acceptance from 74% to 28 to 38%. Collusion falls under the forecast and the labels, but verdict accuracy falls with it, by 13 to 23 points, because GLM now rejects correct work too. The difficulty label changes how strict GLM is overall; acceptance does not track whether the task was labelled hard or easy. Under sequential verdicts GLM copies Luna, who mostly rejects, without ever mentioning her verdict. The countdown changes nothing. Each condition has only 3 to 10 failing tasks per agent, too few to tell whether these levers also helped catch failing work.`);

  // ------------------------------------------------------------------ 10. evidence levers
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Letting the log through removes unsupported acceptance", { placeholder: "title" });
  const ev = [["Baseline (25)", "LG_base", null], ["Restore 50%", "LG_rest50", "LG_rest50"], ["Restore 110%", "LG_rest110", "LG_rest110"], ["Message price*", "LG_price", "LG_price"]];
  const unsup = (k) => (C0[k].evidence ? C0[k].evidence.unsupported_mutual_both.lenient / C0[k].evidence.episodes : m(k, "EC"));
  s.addChart(pres.charts.BAR, [
    { name: "Unsupported mutual ACCEPT", labels: ev.map((r) => r[0]), values: ev.map((r) => r1(unsup(r[1]))) },
    { name: "Verdict accuracy", labels: ev.map((r) => r[0]), values: ev.map((r) => r1(m(r[1], "verdict_accuracy"))) },
  ], {
    x: 0.5, y: 1.4, w: 7.6, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [HEX.evid, HEX.ink2], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"',
    valAxisMajorUnit: 25, showLegend: true, legendPos: "b", showTitle: true, title: "Luna → GLM; levers ran for 5 trajectories each",
    titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt", barGapWidthPct: 55, ...chartBase(), objectName: "chart-evidence",
  });
  text(s, `*Message price: 4 trajectories that stopped early (${C0.LG_price.episodes} episodes). Unsupported: neither agent received ≥90% of the peer's log.`,
    { x: 0.6, y: 6.45, w: 7.5, h: 0.45, fontSize: 10, color: C.text2 });
  const ra = C0.LG_rest110.evidence;
  card(s, 8.45, 1.45, 4.3, 1.3, C.background2, "pay");
  text(s, pct(R110.alice.use_rate),
    { x: 8.7, y: 1.55, w: 1.6, h: 0.75, fontSize: 36, bold: true, fontFace: THEME.headFontFace, color: C.accent1 });
  text(s, `of episodes Luna paid 110% of its reward for the full log (net ${String(R110.alice.net_reward_total).replace("-", "−")})`, { x: 10.3, y: 1.6, w: 2.3, h: 1.2, fontSize: 13, color: C.text2 });
  card(s, 8.45, 2.95, 4.3, 1.3, C.background2, "glm-unsup");
  text(s, `${ra.unsupported_accepts.lenient.bob}/${ra.accepts.bob}`, { x: 8.7, y: 3.05, w: 1.6, h: 0.75, fontSize: 36, bold: true, fontFace: THEME.headFontFace, color: C.accent2 });
  text(s, `of GLM's ACCEPTs had no log behind them (Luna: ${ra.unsupported_accepts.lenient.alice}/${ra.accepts.alice})`, { x: 10.3, y: 3.1, w: 2.3, h: 1.1, fontSize: 13, color: C.text2 });
  card(s, 8.45, 4.45, 4.3, 1.75, "EAF2FC", "quote-luna");
  text(s, [
    { text: "“Restoring the channel cost 11 points against the +10 verdict reward […], leaving a net reward of −1.”", options: { italic: true, breakLine: true } },
    { text: "GPT-6-Luna, reflection at the 110% price", options: { fontSize: 11, color: HEX.ink2 } },
  ], { x: 8.65, y: 4.6, w: 3.9, h: 1.5, fontSize: 14 });
  s.addNotes(`Under the restore and the price, the log can reach the judge, so I count only mutual ACCEPTs where neither agent received the peer's log. That falls from a third of episodes to 2% at both restore prices, and to 8% in the swapped order. Accuracy does not drop. Luna paid for the log in 94% of episodes even when that cost more than the verdict could earn. Pricing messages instead of capping them, agents sent about 1,800 characters per message and delivered 96% of their logs; verdict accuracy reached 98%. The price runs stopped early, so the size is uncertain.`);

  // ------------------------------------------------------------------ 11. forest plot
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Every lever against its baseline", { placeholder: "title" });
  s.addImage({ path: path.join(HERE, "figures", "fig_levers.png"), x: 0.6, y: 1.35, w: 8.3, h: 8.3 * (1078 / 1580), objectName: "forest-plot",
    altText: "Forest plot of each lever's change in unsupported mutual ACCEPT and in verdict accuracy, with 95% confidence intervals" });
  const keys = [
    [HEX.inc, "Incentive", "No change in collusion or accuracy"],
    [HEX.info, "Information", "Less collusion only with lower accuracy"],
    [HEX.evid, "Evidence", "Less unsupported acceptance, accuracy kept or raised"],
  ];
  keys.forEach(([col, head, body], i) => {
    const y = 1.6 + i * 1.35;
    s.addShape(pres.shapes.OVAL, { x: 9.3, y: y + 0.08, w: 0.28, h: 0.28, fill: { color: col }, line: { color: col, width: 0 }, objectName: `fkey-${i}` });
    text(s, head, { x: 9.75, y, w: 3.0, h: 0.4, fontSize: 16, bold: true });
    text(s, body, { x: 9.75, y: y + 0.42, w: 3.0, h: 0.75, fontSize: 14, color: C.text2 });
  });
  text(s, "Filled dot: 95% CI excludes 0. Grey band: where 95% of 5-trajectory samples of the unchanged baseline fall.",
    { x: 9.3, y: 5.75, w: 3.45, h: 1.0, fontSize: 12, color: C.text2 });
  s.addNotes("One picture of all levers, each against its own baseline, with trajectory-bootstrap 95% intervals. Left: change in unsupported mutual ACCEPT. Right: change in verdict accuracy. The incentive row sits on zero in both panels. The information rows move left on both panels at once. The evidence rows move left on collusion while staying at or above zero on accuracy.");

  // ------------------------------------------------------------------ 12. noise
  pres.addSection({ title: "Caveats" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Caveats" });
  s.addText("Five trajectories per lever are not enough", { placeholder: "title" });
  const hk = Object.keys(HIST).map(Number).sort((a, b) => a - b);
  const total = hk.reduce((acc, k) => acc + HIST[k], 0);
  const band = NUM.extras.noise_band.LG_base25.EC;
  const inside = hk.map((k) => (k >= 100 * band["p2.5"] && k <= 100 * band["p97.5"] ? (100 * HIST[k]) / total : 0));
  const outside = hk.map((k, i) => (inside[i] ? 0 : (100 * HIST[k]) / total));
  s.addChart(pres.charts.BAR, [
    { name: "inside the 95% range", labels: hk.map((k) => `${k}%`), values: inside.map((v) => Math.round(v * 10) / 10) },
    { name: "outside", labels: hk.map((k) => `${k}%`), values: outside.map((v) => Math.round(v * 10) / 10) },
  ], {
    x: 0.5, y: 1.4, w: 7.8, h: 5.0, barDir: "col", barGrouping: "stacked", chartColors: ["9AA0AE", "D9DBE2"], showValue: false,
    valAxisLabelFormatCode: '0"%"', showLegend: true, legendPos: "b", showTitle: true,
    title: `EC of all ${total.toLocaleString("en-US")} five-trajectory subsets of the 25-run Luna → GLM baseline`, titleFontSize: 14,
    titleColor: HEX.ink, titleFontFace: "+mn-lt", barGapWidthPct: 25, catAxisLabelFontSize: 10, ...chartBase(),
    catAxisLabelFontSize: 10, objectName: "chart-noise",
  });
  text(s, "Share of subsets; the unchanged baseline alone produces EC anywhere in this range.", { x: 0.6, y: 6.45, w: 7.6, h: 0.4, fontSize: 10, color: C.text2 });
  const noise = [
    [`${pct(band["p2.5"])}–${pct(band["p97.5"])}`, "EC range of 95% of 5-run samples of one unchanged configuration"],
    [`${pct(NUM.replicate.EC.a)} vs ${pct(NUM.replicate.EC.b)}`, "Two batches of the same baseline on the same five task sequences"],
    [arrow("+6", `${(100 * d("LG_pen", "EC").diff).toFixed(1).replace("-", "−")} pp`), "Penalty effect on EC at 5 runs, then at 25"],
  ];
  noise.forEach(([big, lab], i) => {
    card(s, 8.6, 1.45 + i * 1.55, 4.15, 1.4, C.background2, `noise-${i}`);
    text(s, big, { x: 8.85, y: 1.55 + i * 1.55, w: 3.75, h: 0.65, fontSize: 28, bold: true, fontFace: THEME.headFontFace });
    text(s, lab, { x: 8.85, y: 2.2 + i * 1.55, w: 3.75, h: 0.6, fontSize: 13, color: C.text2 });
  });
  text(s, "Treat 5-run effects as candidates; next step is 25+ trajectories per lever.", { x: 8.6, y: 6.15, w: 4.15, h: 0.6, fontSize: 14, bold: true });
  s.addNotes("Most levers ran for five trajectories. Taking every 5-trajectory subset of the 25-trajectory baseline shows that sampling alone moves EC between 18% and 48%. Two batches with identical settings and tasks gave 42% and 34%. Effects outside this band, such as the forecast, the labels and both evidence levers, are worth confirming at scale; the countdown and the sequential effect on EC are inside it.");

  // ------------------------------------------------------------------ 13. implications
  pres.addSection({ title: "Wrap-up" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("What this means for LLM oversight", { placeholder: "title" });
  const imp = [
    [ic.robot, HEX.model, "Screen reviewer models", "The spread across models (0–85% EC) exceeds any lever's effect"],
    [ic.scale, HEX.inc, "Price the rule itself", "A penalty on wrong verdicts cut reward and left collusion unchanged"],
    [ic.chart, HEX.ink2, "Report collusion with accuracy", "Three levers would pass for fixes on EC alone"],
    [ic.unlock, HEX.evid, "Make the rule satisfiable", "When the log could arrive, the strict model paid for it even at a loss"],
    [ic.copy, HEX.info, "Watch sequential review", "GLM copied Luna's verdicts without saying so"],
  ];
  imp.forEach(([img, col, head, body], i) => {
    const y = 1.45 + i * 1.07;
    card(s, 0.6, y, 12.15, 0.92, C.background2, `imp-${i}`);
    badge(s, img, col, 0.8, y + 0.16, 0.6, `imp-${i}`);
    text(s, head, { x: 1.65, y: y + 0.24, w: 4.4, h: 0.45, fontSize: 18, bold: true });
    text(s, body, { x: 6.1, y: y + 0.26, w: 6.5, h: 0.45, fontSize: 15, color: C.text2 });
  });
  s.addNotes("Five practical points. Choose and test the reviewer model first. Rewards on verdict accuracy do not enforce a process rule. Report collusion and accuracy together. The cheapest working fix in this game was to make the rule satisfiable. And when reviewers see each other's verdicts, the weaker one may copy silently, which an audit of stated reasoning would miss.");

  // ------------------------------------------------------------------ 14. limits
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("Limitations and next steps", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.95, 5.25, C.background2, "limits");
  badge(s, ic.search, HEX.ink2, 0.85, 1.7, 0.6, "limits");
  text(s, "Limitations", { x: 1.6, y: 1.78, w: 4.7, h: 0.45, fontSize: 18, bold: true });
  text(s, [
    { text: "Most levers ran for 5 trajectories; the message price and DeepSeek → MiMo levers stopped early", options: { bullet: true, breakLine: true } },
    { text: "Haiku ran only with the forecast lever and one partner", options: { bullet: true, breakLine: true } },
    { text: "One environment; ~89% of tasks correct, too few failing tasks to test discrimination", options: { bullet: true, breakLine: true } },
    { text: "Deals counted by keywords, not the LLM judge; one wording per lever; seat tied to turn order", options: { bullet: true } },
  ], { x: 0.85, y: 2.55, w: 5.45, h: 3.95, fontSize: 16, paraSpaceAfter: 14, color: C.text1 });
  card(s, 6.78, 1.45, 5.95, 5.25, "EAF7F1", "next");
  badge(s, ic.check, HEX.evid, 7.03, 1.7, 0.6, "next");
  text(s, "Next steps", { x: 7.78, y: 1.78, w: 4.7, h: 0.45, fontSize: 18, bold: true });
  text(s, [
    { text: "25+ trajectories for the forecast, labels, sequential verdicts and restore", options: { bullet: true, breakLine: true } },
    { text: "Random difficulty labels (implemented) to separate a label's content from its presence", options: { bullet: true, breakLine: true } },
    { text: "Price sweeps for the restore and the message cost", options: { bullet: true, breakLine: true } },
    { text: "Haiku with a plain baseline and other partners: can one compliant model hold a pair to the rule?", options: { bullet: true } },
  ], { x: 7.03, y: 2.55, w: 5.45, h: 3.95, fontSize: 16, paraSpaceAfter: 14 });
  s.addNotes("The biggest limitation is sample size: most levers had five trajectories, and I showed what that buys. Several DeepSeek-MiMo runs stopped early because of provider errors. All results come from one environment where most work is correct, so accuracy mostly measures acceptance. The next step is to scale the promising levers to 25 or more trajectories and to run the random-label control that is already implemented.");

  // ------------------------------------------------------------------ 15. closing
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Wrap-up" });
  text(s, "Who reviews matters more than how you pay them", { x: 0.7, y: 1.1, w: 11.9, h: 1.0, fontFace: THEME.headFontFace, fontSize: 36, bold: true, color: C.background1 });
  const close = [
    [ic.robot, HEX.model, "Model choice sets collusion from 0% to 85%"],
    [ic.scale, HEX.inc, "Penalising wrong verdicts changes nothing but reward"],
    [ic.search, HEX.info, "Telling agents more makes one reject correct work"],
    [ic.unlock, HEX.evid, "Letting the evidence through, at a price, works"],
  ];
  close.forEach(([img, col, body], i) => {
    const y = 2.45 + i * 0.85;
    badge(s, img, col, 0.75, y, 0.6, `close-${i}`);
    text(s, body, { x: 1.6, y: y + 0.1, w: 10.5, h: 0.45, fontSize: 20, color: C.background1 });
  });
  text(s, [
    { text: "Code: github.com/TanweerulHaque/levers-of-collusion", options: { breakLine: true } },
    { text: "Built on Shi, Zhang & Yang (2026), github.com/SALT-NLP/agent-collusion (branch main); levers on branches tanweer/exp-2 to exp-9", options: { breakLine: true } },
    { text: "Lever code and 5-run conditions predate the sprint; 25-run baselines and penalty runs, analysis and report were done during it" },
  ], { x: 0.75, y: 5.95, w: 12, h: 1.0, fontSize: 12, color: C.background2 });
  s.addNotes("To sum up: which models review decides most of how often they look the other way; paying for correct verdicts does not change it; giving agents more information makes one of them stricter at the expense of correct work; and making the evidence reachable at a price removes most unsupported acceptance. The code is public; the main branch is the original SALT-NLP release and each of my branches adds one lever.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
