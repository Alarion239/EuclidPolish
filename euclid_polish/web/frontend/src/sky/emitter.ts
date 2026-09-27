/* A tiny typed event emitter. Aladin keeps one callback per event and has no
 * `off`, so the engine registers each Aladin event once and fans it out
 * through this; subscribers can come and go. A throwing listener is logged
 * and never stops the others. */

export type EventMap = Record<string, unknown[]>;
type Listener<A extends unknown[]> = (...args: A) => void;

export class Emitter<M extends EventMap> {
  private listeners = new Map<keyof M, Set<Listener<never[]>>>();

  on<K extends keyof M>(event: K, fn: Listener<M[K]>): () => void {
    let set = this.listeners.get(event);
    if (!set) { set = new Set(); this.listeners.set(event, set); }
    set.add(fn as unknown as Listener<never[]>);
    return () => { set!.delete(fn as unknown as Listener<never[]>); };
  }

  emit<K extends keyof M>(event: K, ...args: M[K]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const fn of [...set]) {
      try {
        (fn as unknown as Listener<M[K]>)(...args);
      } catch (err) {
        console.error(`sky: "${String(event)}" listener failed`, err);
      }
    }
  }

  count<K extends keyof M>(event: K): number {
    return this.listeners.get(event)?.size ?? 0;
  }

  clear(): void {
    this.listeners.clear();
  }
}
