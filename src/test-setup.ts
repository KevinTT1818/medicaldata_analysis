/**
 * 全局测试准备。
 *
 * 测试环境没有 matchMedia，而 ChartTheme 靠它判断深浅色。这里装一个可控的桩：
 * 默认浅色，测试可以用 testing/theme.ts 里的 setPrefersDark 切换，
 * 这样两套配色都能测到，而不是只测浅色那一半。
 */
interface MediaListener {
  query: string;
  handler: (e: { matches: boolean }) => void;
}

const listeners: MediaListener[] = [];

declare global {
  // eslint-disable-next-line no-var
  var __prefersDark: boolean;
  // eslint-disable-next-line no-var
  var __notifyColorScheme: (dark: boolean) => void;
}

globalThis.__prefersDark = false;

globalThis.__notifyColorScheme = (dark: boolean) => {
  globalThis.__prefersDark = dark;
  for (const l of listeners) {
    if (l.query.includes('prefers-color-scheme: dark')) l.handler({ matches: dark });
  }
};

globalThis.matchMedia = ((query: string) => ({
  media: query,
  get matches() {
    return query.includes('prefers-color-scheme: dark') ? globalThis.__prefersDark : false;
  },
  onchange: null,
  addEventListener: (_type: string, handler: (e: { matches: boolean }) => void) => {
    listeners.push({ query, handler });
  },
  removeEventListener: (_type: string, handler: (e: { matches: boolean }) => void) => {
    const i = listeners.findIndex((l) => l.handler === handler);
    if (i >= 0) listeners.splice(i, 1);
  },
  addListener: () => {},
  removeListener: () => {},
  dispatchEvent: () => false,
})) as unknown as typeof matchMedia;

// ECharts 在挂载时会取 canvas 上下文。测试只断言 option 对象，不看渲染结果，
// 但没有这个桩会刷一屏 "getContext() not implemented" 把真正的失败淹掉。
if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = (() => ({
    canvas: {},
    measureText: () => ({ width: 0 }),
    createLinearGradient: () => ({ addColorStop: () => {} }),
    createRadialGradient: () => ({ addColorStop: () => {} }),
    createPattern: () => null,
    getImageData: () => ({ data: new Uint8ClampedArray(4) }),
    putImageData: () => {},
    setTransform: () => {},
    save: () => {}, restore: () => {}, scale: () => {}, rotate: () => {}, translate: () => {},
    beginPath: () => {}, closePath: () => {}, moveTo: () => {}, lineTo: () => {},
    bezierCurveTo: () => {}, quadraticCurveTo: () => {}, arc: () => {}, rect: () => {},
    fill: () => {}, stroke: () => {}, clip: () => {}, clearRect: () => {},
    fillRect: () => {}, strokeRect: () => {}, fillText: () => {}, strokeText: () => {},
    drawImage: () => {}, isPointInPath: () => false,
  })) as unknown as HTMLCanvasElement['getContext'];
}

export {};
