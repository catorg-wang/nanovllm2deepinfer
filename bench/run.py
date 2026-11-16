import argparse
import json
import os
from random import Random
from time import perf_counter

from bench.metrics import format_report, summarize_requests, summarize_steps
from bench.trace import TraceWriter
from bench.workloads import SCENARIOS
from nanovllm import SamplingParams
import torch

# engine knobs later sections turn; only the ones given on the command line override Config's defaults
ENGINE_ARGS = {"max_num_seqs": int, "max_num_batched_tokens": int, "gpu_memory_utilization": float, "max_model_len": int}


def drive(llm, requests, seed: int = 0, num_steps: int | None = None) -> float:
    """Feed requests by arrival_step, step the engine until all are done (or num_steps engine steps). Returns wall seconds."""
    t1 = perf_counter()
    rng = Random(seed)
    arrival_step = 0
    engine_step = 0
    i = 0

    while i < len(requests) or not llm.is_finished():
        if num_steps is not None and engine_step >= num_steps:
            break

        while i < len(requests) and requests[i].arrival_step == arrival_step:
            request = requests[i]
            prompt = [rng.randint(0, 10000) for _ in range(request.prompt_len)]
            sp = SamplingParams(max_tokens=request.max_tokens, temperature=request.temperature, ignore_eos=True)
            llm.add_request(prompt=prompt, sampling_params=sp)
            i += 1

        if not llm.is_finished():
            with torch.cuda.nvtx.range(f"engine_step_{engine_step}"):
                llm.step()
            engine_step += 1

        arrival_step += 1

    return perf_counter() - t1

def warmup(llm, requests, n: int = 64):
    """Run a slice of the real workload so torch.compile has seen its shapes before we measure.
    A single short request is not enough: the first big prefill would recompile inside the measurement."""
    
    rng = Random(-1)
    prompts = [[rng.randint(0, 10000) for _ in range(r.prompt_len)] for r in requests[:n]]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)
    # batch size 1 is specialized by torch.compile and needs its own compile (e.g. a lone re-prefill after preemption)
    llm.generate(prompts[:1], sp, use_tqdm=False)
    llm.step_records.clear()
    llm.request_records.clear()


def add_engine_args(p: argparse.ArgumentParser):
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    p.add_argument("--enforce-eager", action="store_true")
    for name, type_ in ENGINE_ARGS.items():
        p.add_argument("--" + name.replace("_", "-"), type=type_)


def make_llm(args):
    from nanovllm import LLM
    kwargs = {"max_model_len": 4096}
    kwargs.update({k: getattr(args, k) for k in ENGINE_ARGS if getattr(args, k) is not None})
    return LLM(os.path.expanduser(args.model), enforce_eager=args.enforce_eager, **kwargs)


def main():
    p = argparse.ArgumentParser()
    add_engine_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, help="stop after this many engine steps (for profiling)")
    p.add_argument("--trace", help="write per-step JSONL here")
    p.add_argument("--json", help="write the summary here")
    args = p.parse_args()

    
    llm = make_llm(args)
    requests = SCENARIOS[args.scenario](args.seed)
    warmup(llm, requests)

    writer = None
    if args.trace:
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
        writer = llm.trace = TraceWriter(args.trace)
    # nsys --capture-range=cudaProfilerApi records only this span, not loading, warmup or compilation
    torch.cuda.profiler.start()
    duration = drive(llm, requests, args.seed, args.num_steps)
    torch.cuda.profiler.stop()
    if writer:
        writer.close()

    steps = summarize_steps(llm.step_records)
    requests = summarize_requests(llm.request_records, duration)
    print(format_report(args.scenario, steps, requests))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"scenario": args.scenario, "steps": steps, "requests": requests}, f, indent=2)


if __name__ == "__main__":
    main()
