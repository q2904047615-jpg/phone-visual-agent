const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const page = fs.readFileSync(
  path.join(__dirname, 'static', 'touch_calibration.html'),
  'utf8'
);
const script = page.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, 'calibration page inline script must exist');

function createPage({width = 393, height = 840, screenWidth = 393, screenHeight = 852} = {}) {
  const elements = new Map();
  const posted = [];
  const root = {};
  const document = {
    documentElement: root,
    fullscreenElement: null,
    webkitFullscreenElement: null,
    webkitCurrentFullScreenElement: null,
    webkitIsFullScreen: false,
    querySelector(selector) {
      if (!elements.has(selector)) {
        elements.set(selector, {
          style: {},
          textContent: '',
          addEventListener() {},
          getBoundingClientRect() {
            return {left: 0, top: 0, width, height};
          },
        });
      }
      return elements.get(selector);
    },
    addEventListener() {},
  };
  const context = vm.createContext({
    console,
    document,
    window: {
      innerWidth: width,
      innerHeight: height,
      visualViewport: {width, height},
      screen: {width: screenWidth, height: screenHeight},
    },
    fetch: async (url, options = {}) => {
      if (options.body) posted.push(JSON.parse(options.body));
      return {
        ok: true,
        async json() {
          return {session_id: 'offline', samples: []};
        },
      };
    },
    setInterval() { return 0; },
    setTimeout,
    clearTimeout,
    Date,
    Promise,
    Set,
  });
  new vm.Script(script, {filename: 'touch_calibration.html'}).runInContext(context);
  return {
    context,
    document,
    root,
    posted,
    evaluate(expression) {
      return vm.runInContext(expression, context);
    },
  };
}

test('prefers and verifies the legacy WebKit fullscreen entry point', async () => {
  const page = createPage();
  let webkitCalls = 0;
  let standardCalls = 0;
  page.root.webkitRequestFullScreen = () => {
    webkitCalls += 1;
    page.document.webkitIsFullScreen = true;
  };
  page.root.requestFullscreen = () => {
    standardCalls += 1;
  };

  const entered = await page.evaluate('tryFullscreenMethods()');

  assert.equal(entered, true);
  assert.equal(webkitCalls, 1);
  assert.equal(standardCalls, 0);
  assert.equal(page.evaluate('fullscreenMethod'), 'webkitRequestFullScreen');
});

test('uses viewport fallback only when both screen ratios meet the threshold', async () => {
  const page = createPage({height: 810, screenHeight: 852});

  await page.evaluate("enterFullscreen({preventDefault() {}})");

  assert.equal(page.evaluate('calibrationMode'), 'viewport_coverage');
  assert.equal(page.evaluate('setup'), false);
  assert.equal(page.evaluate('viewportCoverage.eligible'), true);
  assert.equal(
    page.evaluate('fullscreenError'),
    'browser_fullscreen_api_unavailable'
  );
});

test('blocks without retry when fullscreen and viewport coverage both fail', async () => {
  const page = createPage({height: 700, screenHeight: 852});

  await page.evaluate("enterFullscreen({preventDefault() {}})");

  assert.equal(page.evaluate('calibrationMode'), 'blocked');
  assert.equal(page.evaluate('locked'), true);
  assert.equal(page.evaluate('sequence'), 0);
  assert.match(page.evaluate('fullscreenError'), /fullscreen_api_unavailable/);
  assert.ok(page.posted.some(state => state.phase === 'blocked'));
  assert.match(
    page.document.querySelector('#detail').textContent,
    /fullscreen_api_unavailable/
  );
});

test('fails closed when screen dimensions are unavailable', () => {
  const page = createPage({screenWidth: 0, screenHeight: 0});

  const coverage = page.evaluate('measureViewportCoverage()');

  assert.equal(coverage.eligible, false);
  assert.equal(coverage.width_ratio, 0);
  assert.equal(coverage.height_ratio, 0);
});
