const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

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
