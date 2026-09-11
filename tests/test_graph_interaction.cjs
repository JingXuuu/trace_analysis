const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('depth hover selects nearest point and hides outside the plot', () => {
  const html = fs.readFileSync(path.join(__dirname, '../src/trace_analysis/static/index.html'), 'utf8');
  const source = html.slice(html.indexOf('    function makeDepthProfileInteractive('),
    html.indexOf('    let reuseLoad'));
  const events = {}, appended = [];
  const metadata = {textContent: JSON.stringify({counts: {1: 50, 2: 120, 3: 20},
    bounds: [.5, .2, .4, .6], xlim: [1, 3], ylim: [0, 150]})};
  const svg = {
    querySelector: () => metadata, viewBox: {baseVal: {x: 0, y: 0, width: 1000, height: 500}},
    appendChild: node => appended.push(node), addEventListener: (name, fn) => { events[name] = fn; },
    getScreenCTM: () => ({inverse() { return this; }}),
  };
  const container = {querySelector: () => svg, appendChild: node => appended.push(node),
    getBoundingClientRect: () => ({left: 0, top: 0, width: 1000})};
  const element = () => ({attrs: {}, style: {}, offsetWidth: 170, offsetHeight: 30,
    setAttribute(name, value) { this.attrs[name] = value; }});
  vm.runInNewContext(source + '\nmakeDepthProfileInteractive(container);', {
    container, comma: n => String(n),
    document: {createElementNS: element, createElement: element},
    DOMPoint: class { constructor(x, y) { this.x = x; this.y = y; } matrixTransform() { return this; } },
  });
  const [marker, tooltip] = appended;
  events.pointermove({clientX: 710, clientY: 200});
  assert.equal(tooltip.textContent, 'Depth 2 · 120 nodes');
  assert.equal(marker.attrs.cx, 700);
  assert.equal(marker.attrs.cy, 160);
  assert.equal(marker.attrs.visibility, 'visible');
  events.pointermove({clientX: 890, clientY: 200});
  assert.equal(tooltip.textContent, 'Depth 3 · 20 nodes');
  events.pointermove({clientX: 100, clientY: 200});
  assert.equal(tooltip.hidden, true);
  events.pointermove({clientX: 510, clientY: 200});
  assert.equal(tooltip.textContent, 'Depth 1 · 50 nodes');
  events.pointerleave();
  assert.equal(marker.attrs.visibility, 'hidden');
});

test('route lock ignores hover and unlocks from related nodes or edges', () => {
  const html = fs.readFileSync(path.join(__dirname, '../src/trace_analysis/static/index.html'), 'utf8');
  const source = html.slice(html.indexOf('    function makeGraphInteractive('),
    html.indexOf('    function styleDepthHeaders('));
  function make(name, kind) {
    const classes = new Set();
    const element = {
      classList: {add: (...xs) => xs.forEach(x => classes.add(x)),
        remove: (...xs) => xs.forEach(x => classes.delete(x)), contains: x => classes.has(x)},
      attrs: {}, setAttribute(k, v) { this.attrs[k] = v; },
      querySelector: () => ({textContent: name}),
      closest: selector => selector.startsWith(`g.${kind}`) ? element : null,
    };
    return element;
  }
  const nodes = ['ROOT', 'A', 'B', 'C'].map(n => make(n, 'node'));
  const edges = ['ROOT->A', 'ROOT->B', 'A->C'].map(n => make(n, 'edge'));
  const svg = make('', 'svg'), listeners = {}, frames = new Map();
  let id = 0;
  svg.querySelectorAll = selector => selector.startsWith('g.node') ? nodes : edges;
  svg.addEventListener = (name, fn) => { listeners[name] = fn; };
  const lockStates = [];
  vm.runInNewContext(source + '\nmakeGraphInteractive(svg, onLockChange);', {
    svg, onLockChange: locked => lockStates.push(locked),
    requestAnimationFrame: fn => { frames.set(++id, fn); return id; },
    cancelAnimationFrame: key => frames.delete(key),
  });
  const emit = (name, target, extra = {}) => listeners[name]({target, ...extra});
  const flush = () => { for (const fn of frames.values()) fn(); frames.clear(); };
  const related = node => node.classList.contains('related-node');
  emit('pointerover', nodes[2]); // Pending hover must not override the click.
  emit('click', nodes[1]);
  flush();
  assert.equal(nodes[1].attrs['aria-pressed'], 'true');
  assert.deepEqual(nodes.map(related), [true, true, false, true]);
  emit('pointerover', nodes[2]);
  emit('pointerleave', svg);
  emit('focusin', nodes[2]);
  emit('click', nodes[2]);
  flush();
  assert.deepEqual(nodes.map(related), [true, true, false, true]);
  emit('click', nodes[3]); // Any related node unlocks.
  assert.equal(svg.classList.contains('is-highlighting'), false);
  assert.equal(nodes[1].attrs['aria-pressed'], 'false');
  emit('pointerover', nodes[2]);
  flush();
  assert.deepEqual(nodes.map(related), [true, false, true, false]);
  emit('click', nodes[1]);
  emit('click', edges[2]); // A highlighted edge also unlocks.
  assert.equal(svg.classList.contains('is-locked'), false);
  emit('keydown', nodes[1], {key: 'Enter', preventDefault() {}});
  assert.equal(svg.classList.contains('is-locked'), true);
  emit('keydown', nodes[1], {key: ' ', preventDefault() {}});
  assert.equal(svg.classList.contains('is-locked'), false);
  assert.deepEqual(lockStates, [false, true, false, true, false, true, false]);
});
