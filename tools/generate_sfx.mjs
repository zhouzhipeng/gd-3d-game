import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const sampleRate = 44_100;
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const outputDirectory = join(projectRoot, "assets", "sfx");
mkdirSync(outputDirectory, { recursive: true });

let noiseState = 0x6d2b79f5;
const noise = () => {
  noiseState ^= noiseState << 13;
  noiseState ^= noiseState >>> 17;
  noiseState ^= noiseState << 5;
  return ((noiseState >>> 0) / 0xffffffff) * 2 - 1;
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const smoothstep = (edge0, edge1, value) => {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
};
const burst = (time, start, duration) => {
  const localTime = time - start;
  if (localTime < 0 || localTime >= duration) return 0;
  return Math.sin((Math.PI * localTime) / duration) ** 2;
};
const bell = (time, start, duration, frequency) => {
  const localTime = time - start;
  if (localTime < 0 || localTime >= duration) return 0;
  const envelope = (1 - Math.exp(-localTime * 90)) * Math.exp(-localTime * 5.5);
  return envelope * (
    Math.sin(2 * Math.PI * frequency * localTime)
    + 0.38 * Math.sin(2 * Math.PI * frequency * 2.01 * localTime)
    + 0.16 * Math.sin(2 * Math.PI * frequency * 3.98 * localTime)
  );
};

const writeWav = (name, duration, sampleAt) => {
  const sampleCount = Math.ceil(duration * sampleRate);
  const samples = new Float64Array(sampleCount);
  let peak = 0;

  for (let index = 0; index < sampleCount; index += 1) {
    const time = index / sampleRate;
    const edgeFade = Math.min(1, time / 0.004, (duration - time) / 0.012);
    const sample = sampleAt(time, index) * Math.max(0, edgeFade);
    samples[index] = sample;
    peak = Math.max(peak, Math.abs(sample));
  }

  const gain = peak > 0 ? 0.88 / peak : 1;
  const dataSize = sampleCount * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);

  for (let index = 0; index < sampleCount; index += 1) {
    const value = Math.round(clamp(samples[index] * gain, -1, 1) * 32_767);
    buffer.writeInt16LE(value, 44 + index * 2);
  }

  writeFileSync(join(outputDirectory, name), buffer);
};

writeWav("ui_toggle.wav", 0.18, (time) => {
  const first = burst(time, 0, 0.075);
  const second = burst(time, 0.055, 0.105);
  return (
    0.7 * first * Math.sin(2 * Math.PI * 520 * time)
    + 0.8 * second * Math.sin(2 * Math.PI * 780 * (time - 0.055))
    + 0.08 * noise() * Math.exp(-time * 45)
  );
});

writeWav("weapon_equip.wav", 0.3, (time) => {
  const sweepEnvelope = smoothstep(0, 0.025, time) * (1 - smoothstep(0.2, 0.3, time));
  const sweepPhase = 2 * Math.PI * (360 * time + 1_800 * time * time);
  const metalTail = Math.exp(-time * 12) * (
    Math.sin(2 * Math.PI * 1_860 * time)
    + 0.42 * Math.sin(2 * Math.PI * 2_780 * time)
  );
  return 0.5 * sweepEnvelope * Math.sin(sweepPhase) + 0.36 * metalTail + 0.12 * noise() * sweepEnvelope;
});

let swordNoise = 0;
writeWav("sword_swing.wav", 0.32, (time) => {
  swordNoise = swordNoise * 0.72 + noise() * 0.28;
  const envelope = burst(time, 0.01, 0.3);
  const chirpPhase = 2 * Math.PI * (1_280 * time - 1_550 * time * time);
  return envelope * (0.72 * swordNoise + 0.45 * Math.sin(chirpPhase));
});

let axeNoise = 0;
writeWav("axe_swing.wav", 0.43, (time) => {
  axeNoise = axeNoise * 0.9 + noise() * 0.1;
  const envelope = burst(time, 0.015, 0.4);
  const chirpPhase = 2 * Math.PI * (520 * time - 455 * time * time);
  return envelope * (0.8 * axeNoise + 0.58 * Math.sin(chirpPhase) + 0.18 * Math.sin(chirpPhase * 0.5));
});

writeWav("weapon_impact.wav", 0.27, (time) => {
  const crack = noise() * Math.exp(-time * 48);
  const body = Math.exp(-time * 15) * (
    Math.sin(2 * Math.PI * 172 * time)
    + 0.55 * Math.sin(2 * Math.PI * 346 * time)
    + 0.22 * Math.sin(2 * Math.PI * 812 * time)
  );
  return 0.72 * crack + 0.68 * body;
});

writeWav("skeleton_break.wav", 0.68, (time) => {
  const hits = [
    [0, 0.12, 205],
    [0.095, 0.13, 278],
    [0.205, 0.16, 164],
    [0.355, 0.2, 116],
  ];
  let sample = 0;
  for (const [start, duration, frequency] of hits) {
    const localTime = time - start;
    if (localTime < 0 || localTime >= duration) continue;
    const envelope = Math.exp(-localTime * 24);
    sample += envelope * (
      0.62 * noise()
      + Math.sin(2 * Math.PI * frequency * localTime)
      + 0.34 * Math.sin(2 * Math.PI * frequency * 2.7 * localTime)
    );
  }
  return sample;
});

writeWav("pickup_chime.wav", 0.52, (time) => (
  0.75 * bell(time, 0, 0.34, 659.25)
  + 0.72 * bell(time, 0.085, 0.36, 987.77)
  + 0.62 * bell(time, 0.18, 0.34, 1_318.51)
));

writeWav("quest_fanfare.wav", 1.04, (time) => (
  0.58 * bell(time, 0, 0.5, 523.25)
  + 0.58 * bell(time, 0.15, 0.5, 659.25)
  + 0.58 * bell(time, 0.3, 0.54, 783.99)
  + 0.72 * bell(time, 0.52, 0.52, 1_046.5)
));

console.log(`Generated 8 original SFX in ${outputDirectory}`);
