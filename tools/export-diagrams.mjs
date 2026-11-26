#!/usr/bin/env node
// export-diagrams.mjs: turn Excalidraw scenes into the diagram files our study pages show.
//
// The diagram format is <name>.excalidraw.svg: an SVG with the Excalidraw scene embedded in it,
// and with the hand-drawn font inlined. One file is both the picture and the editable source:
// open it in excalidraw.com (File > Open) or in the VS Code Excalidraw extension, edit, save.
// The rules for drawing one, for people and for AI agents, are in standards/DIAGRAMS.md.
//
//   node tools/export-diagrams.mjs scene.excalidraw [out.excalidraw.svg]   JSON scene -> diagram
//   node tools/export-diagrams.mjs --png diagram.excalidraw.svg [out.png]   diagram -> PNG preview
//   node tools/export-diagrams.mjs --extract diagram.excalidraw.svg [out.excalidraw]   diagram -> JSON
//
// It runs Excalidraw's own exporter in headless Chrome, so the result is exactly what
// excalidraw.com would save. Needs Node 20+, Chrome or Chromium, and internet access the first
// time (it installs playwright-core into ~/.cache/np-diagrams and loads Excalidraw from esm.sh).
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const EXCALIDRAW = "0.18.0";
const PLAYWRIGHT = "1.56.1";
const CACHE = join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "np-diagrams");
const PAYLOAD = "payload-type:application/vnd.excalidraw+json";

function usage(msg) {
  if (msg) console.error(`export-diagrams: ${msg}`);
  console.error("usage: export-diagrams.mjs <scene.excalidraw> [out.excalidraw.svg]\n" +
                "       export-diagrams.mjs --png <diagram.excalidraw.svg> [out.png]\n" +
                "       export-diagrams.mjs --extract <diagram.excalidraw.svg> [out.excalidraw]");
  process.exit(2);
}

async function playwright() {
  const req = createRequire(join(CACHE, "package.json"));
  try {
    return req("playwright-core");
  } catch {
    mkdirSync(CACHE, { recursive: true });
    if (!existsSync(join(CACHE, "package.json"))) writeFileSync(join(CACHE, "package.json"), "{}\n");
    console.error(`export-diagrams: installing playwright-core@${PLAYWRIGHT} into ${CACHE} (first run only)`);
    execFileSync("npm", ["install", "--silent", "--no-audit", "--no-fund", "--prefix", CACHE,
                         `playwright-core@${PLAYWRIGHT}`], { stdio: "inherit" });
    return req("playwright-core");
  }
}

async function withPage(fn) {
  const { chromium } = await playwright();
  let browser;
  for (const opts of [{ channel: "chrome" }, { channel: "chromium" }, {}]) {
    try { browser = await chromium.launch(opts); break; } catch { /* try the next browser */ }
  }
  if (!browser) usage("no Chrome or Chromium found; install Chrome, or run `npx playwright install chromium`");
  try {
    const page = await browser.newPage({ deviceScaleFactor: 2 });
    page.on("pageerror", (e) => console.error("page error:", e.message));
    await page.setContent("<!doctype html><html><body style='margin:0;background:#fff'></body></html>");
    await page.evaluate(async (v) => {
      window.EXCALIDRAW_ASSET_PATH = `https://esm.sh/@excalidraw/excalidraw@${v}/dist/prod/`;
      window.EX = await import(`https://esm.sh/@excalidraw/excalidraw@${v}?bundle&deps=react@18.3.1,react-dom@18.3.1`);
    }, EXCALIDRAW);
    return await fn(page);
  } finally {
    await browser.close();
  }
}

const args = process.argv.slice(2);
const mode = args[0] === "--png" || args[0] === "--extract" ? args.shift() : "--svg";
const input = args[0] && resolve(args[0]);
if (!input || !existsSync(input)) usage(input ? `${args[0]} does not exist` : "no input file");

if (mode === "--svg") {
  if (!input.endsWith(".excalidraw")) usage("the input must be a .excalidraw JSON scene");
  const out = resolve(args[1] || input.replace(/\.excalidraw$/, ".excalidraw.svg"));
  const scene = JSON.parse(readFileSync(input, "utf8"));
  if (scene.type !== "excalidraw" || !Array.isArray(scene.elements)) usage("not an Excalidraw scene");
  const svg = await withPage((page) => page.evaluate(async (scene) => {
    const el = await window.EX.exportToSvg({
      elements: scene.elements, files: scene.files || {},
      appState: { ...(scene.appState || {}), exportBackground: true, exportEmbedScene: true,
                  viewBackgroundColor: scene.appState?.viewBackgroundColor || "#ffffff" },
      exportPadding: 24,
    });
    return el.outerHTML;
  }, scene));
  if (!svg.includes(PAYLOAD)) usage("export did not embed the scene");
  writeFileSync(out, svg + "\n");
  console.log(`wrote ${out}`);
} else {
  if (!input.endsWith(".svg")) usage("the input must be a .excalidraw.svg diagram");
  const svgText = readFileSync(input, "utf8");
  if (!svgText.includes(PAYLOAD)) usage(`${args[0]} has no embedded scene; re-export it with "Embed scene" on`);
  if (mode === "--extract") {
    const out = resolve(args[1] || input.replace(/\.svg$/, ""));
    const json = await withPage((page) => page.evaluate(async (svgText) => {
      const blob = new Blob([svgText], { type: "image/svg+xml" });
      const data = await window.EX.loadFromBlob(blob, null, null);
      return window.EX.serializeAsJSON(data.elements, data.appState, data.files || {}, "local");
    }, svgText));
    writeFileSync(out, JSON.stringify(JSON.parse(json), null, 2) + "\n");
    console.log(`wrote ${out}`);
  } else {
    const out = resolve(args[1] || input.replace(/\.excalidraw\.svg$|\.svg$/, ".png"));
    await withPage(async (page) => {
      await page.setContent(`<!doctype html><html><body style="margin:0;background:#fff">${svgText}</body></html>`);
      await page.locator("svg").first().screenshot({ path: out });
    });
    console.log(`wrote ${out}`);
  }
}
