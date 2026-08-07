import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "VLM Speed Lab — Qwen3-VL 2B",
  description:
    "Reproducible VLM performance iterations with latency, throughput, memory, and quality evidence.",
};

const iterations = [
  {
    id: "00",
    name: "Transformers baseline",
    stack: "BF16 / SDPA / B1",
    status: "ready",
    change: "Control",
    ttft: null,
    throughput: null,
    e2e: null,
    vram: null,
    quality: "Baseline",
    result: "GPU run pending",
  },
  {
    id: "01",
    name: "Visual work budget",
    stack: "10 frames / 938x518",
    status: "measured",
    change: "Sampler + resize",
    ttft: null,
    throughput: null,
    e2e: "0.44s prep",
    vram: null,
    quality: "Input intact",
    result: "11.38x work cut",
  },
  {
    id: "02",
    name: "Flash Attention 2",
    stack: "BF16 / FA2 / B1",
    status: "queued",
    change: "Attention kernel",
    ttft: null,
    throughput: null,
    e2e: null,
    vram: null,
    quality: "Gate pending",
    result: "Next run",
  },
  {
    id: "03",
    name: "Compiled execution",
    stack: "compile / CUDA graph",
    status: "planned",
    change: "Graph capture",
    ttft: null,
    throughput: null,
    e2e: null,
    vram: null,
    quality: "Gate pending",
    result: "Planned",
  },
  {
    id: "04",
    name: "SGLang + FlashInfer",
    stack: "Radix / continuous batch",
    status: "planned",
    change: "Serving runtime",
    ttft: null,
    throughput: null,
    e2e: null,
    vram: null,
    quality: "Gate pending",
    result: "Planned",
  },
  {
    id: "05",
    name: "TensorRT-LLM",
    stack: "IFB / CUDA graph",
    status: "planned",
    change: "NVIDIA runtime",
    ttft: null,
    throughput: null,
    e2e: null,
    vram: null,
    quality: "Gate pending",
    result: "Planned",
  },
];

const qualityTasks = [
  ["Caption", "semantic F1", "≥ 98% of baseline"],
  ["OCR", "normalized edit score", "≥ 99% of baseline"],
  ["Detection", "box mAP / label F1", "≤ 0.5 pt drop"],
  ["Tracking", "IDF1 / HOTA", "≤ 0.5 pt drop"],
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

        <div className="hero-metric" aria-label="Measured visual work reduction">
          <div className="metric-topline"><span>Measured now</span><span className="verified">● VERIFIED</span></div>
          <div className="big-number">11.38<span>×</span></div>
          <div className="metric-label">less frame × pixel work</div>
          <div className="work-bars" aria-hidden="true">
            <div className="work-row"><span>Before</span><i className="bar before" /><b>55.3M</b></div>
            <div className="work-row"><span>After</span><i className="bar after" /><b>4.86M</b></div>
          </div>
          <p>60 × 1280×720 frames → 10 × 938×518 frames in 0.44s</p>
          <div className="honesty-strip">Preprocessing measurement · end-to-end run pending</div>
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
          <p>Green means measured. Amber means the experiment is wired and next in the queue. Empty cells are never replaced with projections.</p>
        </div>

        <div className="run-context" aria-label="Benchmark run context">
          <span><b>Model</b> Qwen3-VL 2B Instruct</span>
          <span><b>Mode</b> Single request</span>
          <span><b>Sample</b> 3 warmups / 30 runs</span>
          <span><b>Decode</b> Temperature 0</span>
        </div>

        <div className="comparison-table-wrap">
          <table className="comparison-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Variant</th>
                <th scope="col">One change</th>
                <th scope="col">TTFT<br /><span>p50</span></th>
                <th scope="col">Output<br /><span>tok/s</span></th>
                <th scope="col">End-to-end<br /><span>p50</span></th>
                <th scope="col">Peak<br /><span>VRAM</span></th>
                <th scope="col">Quality</th>
                <th scope="col">Result</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {iterations.map((item) => (
                <tr className={item.status} key={item.id}>
                  <td className="row-id">{item.id}</td>
                  <th scope="row" className="variant-cell"><strong>{item.name}</strong><span>{item.stack}</span></th>
                  <td>{item.change}</td>
                  <td className="metric-cell">{item.ttft ?? "—"}</td>
                  <td className="metric-cell">{item.throughput ?? "—"}</td>
                  <td className="metric-cell">{item.e2e ?? "—"}</td>
                  <td className="metric-cell">{item.vram ?? "—"}</td>
                  <td className={item.status === "measured" ? "quality-ok" : "muted-cell"}>{item.quality}</td>
                  <td className={item.status === "measured" ? "result-strong" : "muted-cell"}>{item.result}</td>
                  <td><span className={`status status-${item.status}`}>{item.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-notes">
          <span><b>—</b> Not measured; never estimated</span>
          <span><b>01</b> Preprocessing result only, not model inference</span>
          <span><b>Pass</b> Quality must remain within the declared tolerance</span>
        </div>
      </section>

      <section className="quality-section" id="quality">
        <div className="quality-intro">
          <div className="eyebrow light">QUALITY IS A HARD CONSTRAINT</div>
          <h2>Fast and wrong<br />doesn’t ship.</h2>
          <p>A speedup is promoted only after it clears every applicable task gate against the frozen baseline.</p>
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
          <div className="commit-chip">commit <code>3f96127</code></div>
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
        <div><span className="eyebrow light">NEXT ON THE RIG</span><h2>Establish the Qwen3-VL 2B control.</h2></div>
        <div className="next-run-copy"><p>BF16, SDPA, batch 1. Three warmups, thirty measured runs, image caption + OCR + count + structured extraction.</p><a href="https://github.com/gokayfem/ComfyUI_VLM_nodes/tree/main/benchmarks">Open benchmark kit ↗</a></div>
      </section>

      <footer><div className="brand"><span className="brand-mark">VL</span><span>VLM Speed Lab</span></div><p>Built in public. Measured, not marketed.</p><span>ComfyUI VLM Nodes · 2026</span></footer>
    </main>
  );
}
