import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "VLM Speed Lab — Qwen3-VL 2B",
  description:
    "Reproducible VLM performance iterations with latency, throughput, memory, and quality evidence.",
};

const iterations = [
  {
    id: "00",
    name: "Source-resolution control",
    stack: "BF16 · SDPA · dynamic",
    input: "2048×1365",
    tokens: "11,008 vision · 2,770 input",
    status: "measured",
    change: "Frozen control",
    ttft: ["700.3", "723.1"],
    e2e: ["1395.7", "1487.7"],
    throughput: "42.3",
    vram: "4.55",
    speedup: "1.00× / 1.00× / 1.00×",
    quality: "4/4 concepts",
    exact: "10/10",
  },
  {
    id: "01a",
    name: "Medium visual budget",
    stack: "BF16 · SDPA · dynamic",
    input: "672×448",
    tokens: "1,176 vision · 312 input",
    status: "measured",
    change: "Resize only",
    ttft: ["112.8", "122.8"],
    e2e: ["844.0", "881.7"],
    throughput: "42.2",
    vram: "4.04",
    speedup: "6.21× / 1.65× / 1.00×",
    quality: "4/4 concepts",
    exact: "Semantic",
  },
  {
    id: "01b",
    name: "Aggressive visual budget",
    stack: "BF16 · SDPA · dynamic",
    input: "448×299",
    tokens: "504 vision · 144 input",
    status: "measured",
    change: "Resize only",
    ttft: ["88.2", "90.2"],
    e2e: ["774.1", "780.5"],
    throughput: "43.7",
    vram: "4.00",
    speedup: "7.94× / 1.80× / 1.03×",
    quality: "4/4 concepts",
    exact: "Semantic",
  },
  {
    id: "02",
    name: "Compiled execution",
    stack: "BF16 · SDPA · static cache",
    input: "448×299",
    tokens: "504 vision · 144 input",
    status: "measured",
    change: "Cache + compile",
    ttft: ["76.6", "77.3"],
    e2e: ["290.1", "299.7"],
    throughput: "139.4",
    vram: "4.02",
    speedup: "9.14× / 4.81× / 3.30×",
    quality: "4/4 concepts",
    exact: "Exact vs 01b",
  },
  {
    id: "03a",
    name: "Flash Attention 2 isolated",
    stack: "BF16 · FA2 · dynamic",
    input: "448×299",
    tokens: "504 vision · 144 input",
    status: "regression",
    change: "Attention kernel",
    ttft: ["106.6", "114.0"],
    e2e: ["1033.3", "1054.4"],
    throughput: "32.3",
    vram: "4.00",
    speedup: "6.57× / 1.35× / 0.76×",
    quality: "PASS",
    exact: "Exact vs 01b",
  },
  {
    id: "03b",
    name: "FA2 + compiled decode",
    stack: "BF16 · FA2 · static cache",
    input: "448×299",
    tokens: "hit 96-token cap",
    status: "rejected",
    change: "Cache + compile",
    ttft: ["265.0", "273.8"],
    e2e: ["3028.0", "3042.9"],
    throughput: "34.4",
    vram: "4.02",
    speedup: "2.64× / 0.46× / 0.81×",
    quality: "FAIL",
    exact: "Corrupt repeat",
  },
  {
    id: "04",
    name: "Scoped TF32",
    stack: "BF16 · SDPA · static · TF32",
    input: "448×299",
    tokens: "504 vision · 144 input",
    status: "measured",
    change: "FP32 matmul policy",
    ttft: ["74.3", "79.5"],
    e2e: ["274.5", "289.6"],
    throughput: "149.2",
    vram: "4.03",
    speedup: "9.43× / 5.08× / 3.53×",
    quality: "4/4 concepts",
    exact: "Exact vs 01b",
  },
  {
    id: "05",
    name: "SGLang + FlashInfer",
    stack: "Radix · continuous batch",
    input: "Frozen suite",
    tokens: "concurrency sweep",
    status: "planned",
    change: "Serving runtime",
    ttft: null,
    e2e: null,
    throughput: null,
    vram: null,
    speedup: "Not measured",
    quality: "Gate pending",
    exact: "Pending",
  },
  {
    id: "06",
    name: "TensorRT-LLM",
    stack: "IFB · CUDA graph",
    input: "Frozen suite",
    tokens: "concurrency sweep",
    status: "planned",
    change: "NVIDIA runtime",
    ttft: null,
    e2e: null,
    throughput: null,
    vram: null,
    speedup: "Not measured",
    quality: "Gate pending",
    exact: "Pending",
  },
];

const qualityTasks = [
  ["Caption facts", "required concepts", "4 / 4 in every run"],
  ["Resize fidelity", "task rubric", "pass · wording changed"],
  ["Compiler fidelity", "SHA-256 output", "exact vs iteration 01b"],
  ["Repeatability", "within variant", "10 / 10 identical"],
];

export default function Home() {
  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="VLM Speed Lab home">
          <span className="brand-mark">VL</span>
          <span>VLM Speed Lab</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#iterations">Iterations</a>
          <a href="#quality">Quality gate</a>
          <a href="#protocol">Protocol</a>
        </nav>
        <a className="repo-link" href="https://github.com/gokayfem/ComfyUI_VLM_nodes">
          View repository ↗
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="live-dot" /> Experiment 001 · Qwen3-VL 2B Instruct</div>
          <h1>Make it faster.<br /><em>Prove</em> it stayed good.</h1>
          <p className="lede">
            One model. One frozen test set. One change per iteration. Every speed claim ships with its output, configuration, and quality score.
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="#iterations">Explore the iterations <span>↓</span></a>
            <span className="artifact-note">No synthetic leaderboard numbers</span>
          </div>
        </div>

        <div className="hero-metric" aria-label="Measured end-to-end speedup">
          <div className="metric-topline"><span>Measured now</span><span className="verified">● VERIFIED</span></div>
          <div className="big-number">5.08<span>×</span></div>
          <div className="metric-label">faster end to end</div>
          <div className="work-bars" aria-hidden="true">
            <div className="work-row"><span>Before</span><i className="bar before" /><b>1395.7</b></div>
            <div className="work-row"><span>After</span><i className="bar after" /><b>274.5</b></div>
          </div>
          <p>Milliseconds p50 · 10 measured runs · output throughput 42.3 → 149.2 tok/s</p>
          <div className="honesty-strip">RTX 3090 · batch 1 · task rubric passed · raw samples attached</div>
        </div>
      </section>

      <section className="manifesto-band" aria-label="Benchmark principles">
        <span>01 / Same checkpoint</span>
        <span>02 / Same media</span>
        <span>03 / Same decode</span>
        <span>04 / Quality gated</span>
        <span>05 / Raw artifacts</span>
      </section>

      <section className="section iterations-section" id="iterations">
        <div className="section-heading">
          <div>
            <div className="eyebrow">THE OPTIMIZATION LOG</div>
            <h2>Every millisecond has a paper trail.</h2>
          </div>
          <p>Primary numbers are p50; the smaller number is p95. Every row keeps input work, memory, speedup, and quality evidence in view.</p>
        </div>

        <div className="run-context" aria-label="Benchmark run context">
          <span><b>Model</b> Qwen3-VL 2B Instruct</span>
          <span><b>Mode</b> Single request</span>
          <span><b>Sample</b> 10 measured / variant</span>
          <span><b>Warmup</b> 2 dynamic / 6 compiled</span>
          <span><b>Runtime</b> Torch 2.8 · TF 5.12.1</span>
        </div>

        <div className="comparison-table-wrap">
          <table className="comparison-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Variant</th>
                <th scope="col">Input work</th>
                <th scope="col">One change</th>
                <th scope="col">TTFT<br /><span>p50 / p95 ms</span></th>
                <th scope="col">E2E<br /><span>p50 / p95 ms</span></th>
                <th scope="col">Output<br /><span>tok/s</span></th>
                <th scope="col">Peak<br /><span>VRAM GiB</span></th>
                <th scope="col">Speedup<br /><span>TTFT / E2E / tok/s</span></th>
                <th scope="col">Quality</th>
                <th scope="col">Output fidelity</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {iterations.map((item) => (
                <tr className={item.status} key={item.id}>
                  <td className="row-id">{item.id}</td>
                  <th scope="row" className="variant-cell"><strong>{item.name}</strong><span>{item.stack}</span></th>
                  <td className="input-cell"><strong>{item.input}</strong><span>{item.tokens}</span></td>
                  <td>{item.change}</td>
                  <td className="metric-cell">{item.ttft ? <><strong>{item.ttft[0]}</strong><span>{item.ttft[1]}</span></> : "—"}</td>
                  <td className="metric-cell">{item.e2e ? <><strong>{item.e2e[0]}</strong><span>{item.e2e[1]}</span></> : "—"}</td>
                  <td className="metric-cell">{item.throughput ?? "—"}</td>
                  <td className="metric-cell">{item.vram ?? "—"}</td>
                  <td className={item.status === "measured" ? "speedup-cell" : item.status === "planned" ? "muted-cell" : "regression-cell"}>{item.speedup}</td>
                  <td className={item.status === "rejected" ? "quality-fail" : item.status === "planned" ? "muted-cell" : "quality-ok"}>{item.quality}</td>
                  <td className={item.status === "rejected" ? "quality-fail" : item.status === "planned" ? "muted-cell" : "fidelity-cell"}>{item.exact}</td>
                  <td><span className={`status status-${item.status}`}>{item.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-notes">
          <span><b>—</b> Not measured; never estimated</span>
          <span><b>Semantic</b> Required facts pass; wording changed</span>
          <span><b>Exact</b> SHA-256-identical generated text</span>
          <span><b>Load</b> 88.351s → 6.858s warm cache</span>
        </div>
      </section>

      <section className="quality-section" id="quality">
        <div className="quality-intro">
          <div className="eyebrow light">QUALITY IS A HARD CONSTRAINT</div>
          <h2>Fast and wrong<br />doesn’t ship.</h2>
          <p>A speedup is promoted only after it clears its declared task gate. Semantic preservation and exact bytes are reported separately.</p>
          <div className="gate-formula"><span>promotion rule</span><code>speed ↑ &amp;&amp; quality ≥ tolerance</code></div>
        </div>
        <div className="quality-table" role="table" aria-label="Quality thresholds">
          <div className="quality-row header" role="row"><span>Capability</span><span>Primary score</span><span>Pass threshold</span></div>
          {qualityTasks.map(([task, metric, threshold]) => (
            <div className="quality-row" role="row" key={task}><strong>{task}</strong><span>{metric}</span><b>{threshold}</b></div>
          ))}
          <div className="quality-proof">
            <span className="proof-icon">✓</span>
            <div><strong>Outputs stay attached</strong><p>Prompts, model text, boxes, masks, tracks, timing traces, and environment metadata live beside each result.</p></div>
          </div>
        </div>
      </section>

      <section className="section protocol-section" id="protocol">
        <div className="section-heading protocol-heading">
          <div><div className="eyebrow">REPRODUCIBLE BY DEFAULT</div><h2>The benchmark contract.</h2></div>
          <div className="commit-chip">artifact <code>TF5 · RTX3090 · B1</code></div>
        </div>
        <div className="protocol-grid">
          <article><span>1</span><h3>Freeze</h3><p>Checkpoint revision, media hashes, prompts, seed, precision, and generation parameters.</p></article>
          <article><span>2</span><h3>Warm</h3><p>Cold start is recorded once. Warmups are declared and excluded from steady-state percentiles.</p></article>
          <article><span>3</span><h3>Measure</h3><p>TTFT, inter-token latency, output tokens/sec, end-to-end time, peak VRAM, and concurrency.</p></article>
          <article><span>4</span><h3>Gate</h3><p>Compare outputs to the baseline and ground truth. Publish pass, regression, or inconclusive.</p></article>
        </div>
        <div className="metric-strip">
          <div><small>Latency</small><strong>p50 / p95 / p99</strong></div>
          <div><small>Throughput</small><strong>output tok/s</strong></div>
          <div><small>Responsiveness</small><strong>TTFT + ITL</strong></div>
          <div><small>Efficiency</small><strong>GB VRAM / request</strong></div>
          <div><small>Quality</small><strong>task-specific score</strong></div>
        </div>
      </section>

      <section className="next-run">
        <div><span className="eyebrow light">NEXT ON THE RIG</span><h2>Take 274 ms into SGLang.</h2></div>
        <div className="next-run-copy"><p>Flash Attention was measured and rejected. Next: the same frozen input through SGLang and FlashInfer, followed by concurrency sweeps and broader OCR, count, and extraction coverage.</p><a href="https://github.com/gokayfem/ComfyUI_VLM_nodes/tree/codex/vlm-benchmark-lab/benchmarks">Open benchmark kit ↗</a></div>
      </section>

      <footer><div className="brand"><span className="brand-mark">VL</span><span>VLM Speed Lab</span></div><p>Built in public. Measured, not marketed.</p><span>ComfyUI VLM Nodes · 2026</span></footer>
    </main>
  );
}
