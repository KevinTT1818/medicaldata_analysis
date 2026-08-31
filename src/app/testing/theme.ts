/** 在测试里切换系统配色方案。见 src/test-setup.ts 里的 matchMedia 桩。 */
export function setPrefersDark(dark: boolean): void {
  (globalThis as unknown as { __notifyColorScheme: (d: boolean) => void })
    .__notifyColorScheme(dark);
}

/** 每个测试跑完复位，免得互相影响。 */
export function resetColorScheme(): void {
  setPrefersDark(false);
}
