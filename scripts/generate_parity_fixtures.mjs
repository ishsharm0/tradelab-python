#!/usr/bin/env node

import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SEED = 42;
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const PYTHON_ROOT = resolve(SCRIPT_DIR, "..");
const JS_ROOT = resolve(PYTHON_ROOT, "..", "tradelab");
const FIXTURE_DIR = resolve(PYTHON_ROOT, "tests", "parity", "fixtures");

const sourcePaths = {
  ta: "src/ta/index.js",
  finite: "src/metrics/finite.js",
  annualize: "src/metrics/annualize.js",
  metrics: "src/metrics/buildMetrics.js",
  research: "src/research/index.js",
  backtest: "src/engine/backtest.js",
  ticks: "src/engine/backtestTicks.js",
  financing: "src/engine/execution.js",
  walkForward: "src/engine/walkForward.js",
  portfolio: "src/engine/portfolio.js",
};

const fixtures = {
  ta: "ta.json",
  metrics: "metrics.json",
  research: "research.json",
  backtest: "backtest.json",
  ticks: "ticks.json",
  financing: "financing.json",
  walkForward: "walkForward.json",
  portfolio: "portfolio.json",
};

const importSource = async (sourcePath) => import(pathToFileURL(resolve(JS_ROOT, sourcePath)).href);

const [ta, finite, annualize, metrics, research, barEngine, tickEngine, execution, walkForward, portfolio] =
  await Promise.all([
    importSource(sourcePaths.ta),
    importSource(sourcePaths.finite),
    importSource(sourcePaths.annualize),
    importSource(sourcePaths.metrics),
    importSource(sourcePaths.research),
    importSource(sourcePaths.backtest),
    importSource(sourcePaths.ticks),
    importSource(sourcePaths.financing),
    importSource(sourcePaths.walkForward),
    importSource(sourcePaths.portfolio),
  ]);

function normalizeJson(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return value.map(normalizeJson);
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, normalizeJson(value[key])])
    );
  }
  return null;
}

async function writeFixture(filename, payload) {
  await writeFile(resolve(FIXTURE_DIR, filename), `${JSON.stringify(normalizeJson(payload))}\n`, "utf8");
}

const BASE_TIME = Date.UTC(2024, 0, 2, 14, 30, 0);
const HOUR = 60 * 60 * 1000;
const candles = [
  [100, 102, 99, 101, 120],
  [101, 104, 100, 103, 130],
  [103, 105, 101, 102, 140],
  [102, 106, 102, 105, 150],
  [105, 107, 103, 104, 110],
  [104, 108, 103, 107, 160],
  [107, 109, 105, 106, 170],
  [106, 110, 105, 109, 180],
  [109, 111, 107, 108, 190],
  [108, 112, 107, 111, 200],
  [111, 113, 109, 110, 210],
  [110, 114, 109, 113, 220],
  [113, 115, 111, 112, 230],
  [112, 116, 111, 115, 240],
  [115, 117, 113, 114, 250],
  [114, 118, 113, 117, 260],
  [117, 119, 115, 116, 270],
  [116, 120, 115, 119, 280],
  [119, 121, 117, 118, 290],
  [118, 122, 117, 121, 300],
  [121, 123, 119, 120, 310],
  [120, 124, 119, 123, 320],
  [123, 125, 121, 122, 330],
  [122, 126, 121, 125, 340],
].map(([open, high, low, close, volume], index) => ({
  time: BASE_TIME + index * HOUR,
  open,
  high,
  low,
  close,
  volume,
}));
const closes = candles.map(({ close }) => close);

const tradeClosed = [
  {
    side: "long",
    entry: 100,
    entryFill: 100,
    openTime: BASE_TIME,
    _initRisk: 2,
    exit: { price: 104, time: BASE_TIME + HOUR, reason: "TP", pnl: 40 },
  },
  {
    side: "short",
    entry: 105,
    entryFill: 105,
    openTime: BASE_TIME + HOUR,
    _initRisk: 2,
    exit: { price: 107, time: BASE_TIME + 3 * HOUR, reason: "SL", pnl: -20 },
  },
];

const barScenario = {
  candles: candles.slice(0, 8),
  equity: 10_000,
  warmupBars: 1,
  flattenAtClose: false,
  slippageBps: 0,
  feeBps: 0,
  scaleOutAtR: 0,
  finalTP_R: 2,
  signal: ({ index }) =>
    index === 1 ? { side: "long", entry: 102, stop: 100, takeProfit: 104, qty: 2 } : null,
};
const barResult = barEngine.backtest(barScenario);

const ticks = [100, 100.5, 103, 102].map((price, index) => ({
  time: BASE_TIME + index * 60_000,
  price,
  size: 10 + index,
}));
const tickResult = tickEngine.backtestTicks({
  ticks,
  symbol: "TICK",
  equity: 10_000,
  slippageBps: 0,
  feeBps: 0,
  seed: SEED,
  queueFillProbability: 1,
  signal: ({ index }) =>
    index === 0 ? { side: "long", entry: 100.5, stop: 99, takeProfit: 102.5, qty: 3 } : null,
});

const walkForwardCandles = candles.slice(0, 18);
const walkForwardResult = walkForward.walkForwardOptimize({
  candles: walkForwardCandles,
  parameterSets: [{ target: 1 }, { target: 2 }],
  trainBars: 6,
  testBars: 4,
  stepBars: 4,
  scoreBy: "totalPnL",
  backtestOptions: {
    equity: 10_000,
    warmupBars: 1,
    flattenAtClose: false,
    slippageBps: 0,
    feeBps: 0,
    scaleOutAtR: 0,
  },
  signalFactory: ({ target }) => ({ index, bar }) =>
    index === 1
      ? { side: "long", entry: bar.close, stop: bar.close - 1, takeProfit: bar.close + target, qty: 1 }
      : null,
});

const portfolioResult = portfolio.backtestPortfolio({
  equity: 10_000,
  collectReplay: true,
  processingOrder: "shuffle",
  shuffleSeed: SEED,
  systems: [
    {
      symbol: "ALPHA",
      candles: candles.slice(0, 7),
      warmupBars: 1,
      flattenAtClose: false,
      slippageBps: 0,
      feeBps: 0,
      scaleOutAtR: 0,
      signal: ({ index }) =>
        index === 1 ? { side: "long", entry: 102, stop: 100, takeProfit: 104, qty: 2 } : null,
    },
    {
      symbol: "BETA",
      candles: candles.slice(0, 7).map((bar) => ({ ...bar, close: bar.close + 10, high: bar.high + 10, low: bar.low + 10, open: bar.open + 10 })),
      warmupBars: 1,
      flattenAtClose: false,
      slippageBps: 0,
      feeBps: 0,
      scaleOutAtR: 0,
      signal: ({ index }) =>
        index === 1 ? { side: "short", entry: 113, stop: 115, takeProfit: 111, qty: 2 } : null,
    },
  ],
});

await mkdir(FIXTURE_DIR, { recursive: true });
await writeFixture(fixtures.ta, {
  input: {
    candles,
    closes,
    calls: {
      ema: { period: 5 },
      atr: { period: 5 },
      rsi: { period: 5 },
      macd: { fast: 4, slow: 8, signalPeriod: 3 },
      stochastic: { kPeriod: 5, dPeriod: 3 },
      bollinger: { period: 5, mult: 2 },
      donchian: { period: 5 },
      keltner: { emaPeriod: 5, atrPeriod: 5, mult: 2 },
      supertrend: { period: 5, mult: 2 },
      vwap: {},
      swingHighAt9: { index: 9, left: 2, right: 2 },
      swingLowAt8: { index: 8, left: 2, right: 2 },
      fvgAt3: { index: 3 },
      lastSwingAt12: { index: 12, direction: "down" },
      structureAt12: { index: 12 },
    },
  },
  output: {
    ema: ta.ema(closes, 5),
    atr: ta.atr(candles, 5),
    rsi: ta.rsi(closes, 5),
    macd: ta.macd(closes, 4, 8, 3),
    stochastic: ta.stochastic(candles, 5, 3),
    bollinger: ta.bollinger(closes, 5, 2),
    donchian: ta.donchian(candles, 5),
    keltner: ta.keltner(candles, 5, 5, 2),
    supertrend: ta.supertrend(candles, 5, 2),
    vwap: ta.vwap(candles),
    swingHighAt9: ta.swingHigh(candles, 9),
    swingLowAt8: ta.swingLow(candles, 8),
    fvgAt3: ta.detectFVG(candles, 3),
    lastSwingAt12: ta.lastSwing(candles, 12, "down"),
    structureAt12: ta.structureState(candles, 12),
  },
});
await writeFixture(fixtures.metrics, {
  input: {
    finite: {
      calls: [
        { value: "Infinity", fallback: 0 },
        { value: "-Infinity", fallback: 0 },
        { value: "NaN", fallback: 7 },
      ],
    },
    annualize: {
      calls: [
        { interval: "1d", estBarMs: null },
        { interval: "1m", estBarMs: null },
        { interval: "custom", estBarMs: HOUR },
      ],
    },
    buildMetrics: {
      closed: tradeClosed,
      equityStart: 10_000,
      equityFinal: 10_020,
      candles: candles.slice(0, 4),
      estBarMs: HOUR,
      interval: "1h",
    },
  },
  output: {
    bigNumber: finite.BIG_NUMBER,
    clampFinite: [finite.clampFinite(Infinity), finite.clampFinite(-Infinity), finite.clampFinite(NaN, 7)],
    periodsPerYear: {
      daily: annualize.periodsPerYear("1d"),
      minute: annualize.periodsPerYear("1m"),
      estimated: annualize.periodsPerYear("custom", HOUR),
    },
    buildMetrics: metrics.buildMetrics({
      closed: tradeClosed,
      equityStart: 10_000,
      equityFinal: 10_020,
      candles: candles.slice(0, 4),
      estBarMs: HOUR,
      interval: "1h",
    }),
  },
});
await writeFixture(fixtures.research, {
  input: {
    stats: { normalCdf: 1.25, normalPpf: 0.9, moments: [1, 2, 4, 8, 16] },
    monteCarlo: {
      tradePnls: [10, -5, 7, -3, 4, 6],
      equityStart: 1_000,
      iterations: 32,
      blockSize: 2,
      seed: SEED,
    },
    deflatedSharpe: {
      sharpe: 1.4,
      sampleSize: 64,
      numTrials: 12,
      sharpeStd: 0.3,
      skew: -0.2,
      kurtosis: 3.4,
    },
    pbo: {
      performanceMatrix: [
        [0.1, 0.2, -0.1, 0.3, 0.05, 0.2],
        [0.15, -0.1, 0.2, 0.1, 0.12, -0.05],
        [-0.05, 0.1, 0.05, -0.1, 0.2, 0.1],
      ],
      groups: 4,
    },
    cpcv: { nObservations: 12, nGroups: 4, nTestGroups: 2, embargo: 1 },
  },
  output: {
    stats: { normalCdf: research.normalCdf(1.25), normalPpf: research.normalPpf(0.9), moments: research.moments([1, 2, 4, 8, 16]) },
    monteCarlo: research.monteCarlo({ tradePnls: [10, -5, 7, -3, 4, 6], equityStart: 1_000, iterations: 32, blockSize: 2, seed: SEED }),
    deflatedSharpe: research.deflatedSharpe({ sharpe: 1.4, sampleSize: 64, numTrials: 12, sharpeStd: 0.3, skew: -0.2, kurtosis: 3.4 }),
    pbo: research.probabilityOfBacktestOverfitting([[0.1, 0.2, -0.1, 0.3, 0.05, 0.2], [0.15, -0.1, 0.2, 0.1, 0.12, -0.05], [-0.05, 0.1, 0.05, -0.1, 0.2, 0.1]], { groups: 4 }),
    cpcv: research.combinatorialPurgedSplits({ nObservations: 12, nGroups: 4, nTestGroups: 2, embargo: 1 }),
  },
});
await writeFixture(fixtures.backtest, {
  input: {
    options: {
      candles: barScenario.candles,
      equity: 10_000,
      warmupBars: 1,
      flattenAtClose: false,
      slippageBps: 0,
      feeBps: 0,
      scaleOutAtR: 0,
      finalTP_R: 2,
    },
    signal: {
      kind: "index-equals",
      index: 1,
      value: { side: "long", entry: 102, stop: 100, takeProfit: 104, qty: 2 },
    },
  },
  output: barResult,
});
await writeFixture(fixtures.ticks, {
  input: {
    options: {
      ticks,
      symbol: "TICK",
      equity: 10_000,
      slippageBps: 0,
      feeBps: 0,
      seed: SEED,
      queueFillProbability: 1,
    },
    signal: {
      kind: "index-equals",
      index: 0,
      value: { side: "long", entry: 100.5, stop: 99, takeProfit: 102.5, qty: 3 },
    },
  },
  output: tickResult,
});
await writeFixture(fixtures.financing, {
  input: {
    fundingEvents: { fromMs: BASE_TIME, toMs: BASE_TIME + 25 * HOUR, intervalMs: 8 * HOUR, anchorMs: 0 },
    long: {
      side: "long",
      notional: 10_000,
      fromMs: BASE_TIME,
      toMs: BASE_TIME + 25 * HOUR,
      costs: {
        carry: { longAnnualBps: 365, shortAnnualBps: 120 },
        funding: { intervalMs: 8 * HOUR, rateBps: 1.5, anchorMs: 0 },
      },
    },
    short: {
      side: "short",
      notional: 10_000,
      fromMs: BASE_TIME,
      toMs: BASE_TIME + 25 * HOUR,
      costs: {
        carry: { longAnnualBps: 365, shortAnnualBps: 120 },
        funding: { intervalMs: 8 * HOUR, rateBps: 1.5, anchorMs: 0 },
      },
    },
  },
  output: {
    fundingEvents: execution.fundingEvents(BASE_TIME, BASE_TIME + 25 * HOUR, 8 * HOUR, 0),
    long: execution.financingCost({ side: "long", notional: 10_000, fromMs: BASE_TIME, toMs: BASE_TIME + 25 * HOUR, costs: { carry: { longAnnualBps: 365, shortAnnualBps: 120 }, funding: { intervalMs: 8 * HOUR, rateBps: 1.5, anchorMs: 0 } } }),
    short: execution.financingCost({ side: "short", notional: 10_000, fromMs: BASE_TIME, toMs: BASE_TIME + 25 * HOUR, costs: { carry: { longAnnualBps: 365, shortAnnualBps: 120 }, funding: { intervalMs: 8 * HOUR, rateBps: 1.5, anchorMs: 0 } } }),
  },
});
await writeFixture(fixtures.walkForward, {
  input: {
    options: {
      candles: walkForwardCandles,
      parameterSets: [{ target: 1 }, { target: 2 }],
      trainBars: 6,
      testBars: 4,
      stepBars: 4,
      scoreBy: "totalPnL",
      backtestOptions: {
        equity: 10_000,
        warmupBars: 1,
        flattenAtClose: false,
        slippageBps: 0,
        feeBps: 0,
        scaleOutAtR: 0,
      },
    },
    signalFactory: {
      kind: "index-equals-from-bar",
      index: 1,
      value: { side: "long", entry: "bar.close", stop: "bar.close - 1", takeProfit: "bar.close + params.target", qty: 1 },
    },
  },
  output: walkForwardResult,
});
await writeFixture(fixtures.portfolio, {
  input: {
    options: {
      equity: 10_000,
      collectReplay: true,
      processingOrder: "shuffle",
      shuffleSeed: SEED,
      systems: [
        {
          symbol: "ALPHA",
          candles: candles.slice(0, 7),
          warmupBars: 1,
          flattenAtClose: false,
          slippageBps: 0,
          feeBps: 0,
          scaleOutAtR: 0,
          signal: {
            kind: "index-equals",
            index: 1,
            value: { side: "long", entry: 102, stop: 100, takeProfit: 104, qty: 2 },
          },
        },
        {
          symbol: "BETA",
          candles: candles.slice(0, 7).map((bar) => ({ ...bar, close: bar.close + 10, high: bar.high + 10, low: bar.low + 10, open: bar.open + 10 })),
          warmupBars: 1,
          flattenAtClose: false,
          slippageBps: 0,
          feeBps: 0,
          scaleOutAtR: 0,
          signal: {
            kind: "index-equals",
            index: 1,
            value: { side: "short", entry: 113, stop: 115, takeProfit: 111, qty: 2 },
          },
        },
      ],
    },
  },
  output: portfolioResult,
});

const packageJson = JSON.parse(await readFile(resolve(JS_ROOT, "package.json"), "utf8"));
await writeFixture("manifest.json", {
  sourceVersion: packageJson.version,
  seed: SEED,
  fixtures,
  sourcePaths: Object.fromEntries(
    Object.entries(sourcePaths).map(([name, sourcePath]) => [name, relative(JS_ROOT, resolve(JS_ROOT, sourcePath))])
  ),
});
