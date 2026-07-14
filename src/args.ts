/**
 * Tiny flag parser: `--flag value`, `--flag=value`, boolean flags, positionals.
 * Unknown flags are usage errors — a calculator that silently ignores a typoed
 * `--kv-dytpe` would hand out wrong numbers with a straight face.
 */

export class UsageError extends Error {}

export interface FlagSpec {
  /** Flag name without the leading dashes, e.g. "ctx". */
  name: string;
  /** Whether the flag consumes a value. */
  takesValue: boolean;
}

export interface ParsedArgs {
  values: Map<string, string>;
  flags: Set<string>;
  positionals: string[];
}

export function parseArgs(argv: string[], specs: FlagSpec[]): ParsedArgs {
  const byName = new Map(specs.map((s) => [s.name, s]));
  const values = new Map<string, string>();
  const flags = new Set<string>();
  const positionals: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (!arg.startsWith("--")) {
      positionals.push(arg);
      continue;
    }
    const eq = arg.indexOf("=");
    const name = eq === -1 ? arg.slice(2) : arg.slice(2, eq);
    const spec = byName.get(name);
    if (!spec) throw new UsageError(`unknown flag --${name}`);
    if (!spec.takesValue) {
      if (eq !== -1) throw new UsageError(`flag --${name} does not take a value`);
      flags.add(name);
      continue;
    }
    let value: string;
    if (eq !== -1) {
      value = arg.slice(eq + 1);
    } else {
      const next = argv[i + 1];
      if (next === undefined) throw new UsageError(`flag --${name} requires a value`);
      value = next;
      i++;
    }
    if (values.has(name)) throw new UsageError(`flag --${name} given twice`);
    values.set(name, value);
  }

  return { values, flags, positionals };
}

/** Parse a positive integer flag value. */
export function positiveInt(name: string, value: string): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1) {
    throw new UsageError(`--${name} must be a positive integer, got "${value}"`);
  }
  return n;
}
