import argparse
import json
import sys
import time
from pathlib import Path

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = ROOT / "docs/minimind-moe-baseline"
WORKLOADS = ((128, 128, 1), (128, 128, 8), (768, 32, 1), (32, 256, 8))


def load_cases(baseline_dir: Path):
    data = json.loads((baseline_dir / "inputs.json").read_text(encoding="utf-8"))
    return {case["name"]: case["input_ids"] for case in data["token_cases"] + data["chat_cases"]}


def create_llm(model: str, scenario: str):
    return LLM(
        model,
        dtype=torch.float16,
        attention_backend="fa",
        moe_backend="fused",
        cache_type="radix" if scenario == "radix" else "naive",
        cuda_graph_max_bs=8 if scenario in ("graph", "perf") else 0,
        max_extend_tokens=128 if scenario == "chunk" else 8192,
        max_seq_len_override=2048,
        max_running_req=8,
        num_page_override=8192,
    )


def generate(llm: LLM, prompts, max_tokens=32, ignore_eos=False):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos)
    return [result["token_ids"] for result in llm.generate(prompts, params)]


def create_reference_model(runtime: str, model_path: str):
    if runtime == "transformers":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16
        ).cuda().eval()

    minimind_root = Path(model_path).parents[2]
    sys.path.insert(0, str(minimind_root))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=True)
    )
    checkpoint = minimind_root / "out" / f"{Path(model_path).name}.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.half().cuda().eval()


def generate_reference(model, prompts, max_tokens):
    input_ids = torch.tensor(prompts, device="cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=True,
        )
    return output[:, input_ids.shape[1] :].tolist()


def run_correctness(llm: LLM, scenario: str, cases, reference, baseline_dir: Path):
    if scenario == "base":
        names = ["ids_32", "ids_128", "ids_768", "chat_single", "chat_multi"]
        outputs = {name: generate(llm, [cases[name]])[0] for name in names}
        hf = torch.load(
            baseline_dir / "transformers_baseline.pt", map_location="cpu", weights_only=True
        )["sft"]
        hf_outputs = {name: hf[name]["output_ids"].tolist() for name in names}
        first_token_match = {
            name: outputs[name][:1] == hf_outputs[name][:1] for name in names
        }
        full_match = {name: outputs[name] == hf_outputs[name] for name in names}
        return {
            "passed": all(full_match.values()),
            "first_token_passed": all(first_token_match.values()),
            "outputs": outputs,
            "transformers_first_token_match": first_token_match,
            "transformers_full_match": full_match,
        }

    expected = json.loads(Path(reference).read_text(encoding="utf-8"))["outputs"]
    if scenario == "chunk":
        output = generate(llm, [cases["ids_768"]])[0]
        repeated = generate(llm, [cases["ids_768"]])[0]
        matches = {"base": output == expected["ids_768"], "repeat": repeated == output}
        return {
            "passed": all(matches.values()),
            "outputs": {"ids_768": output},
            "match": matches,
        }

    if scenario == "radix":
        ids_128 = generate(llm, [cases["ids_128"]])[0]
        ids_768 = generate(llm, [cases["ids_768"]])[0]
        ids_768_repeat = generate(llm, [cases["ids_768"]])[0]
        matches = {
            "ids_128": ids_128 == expected["ids_128"],
            "ids_768": ids_768 == expected["ids_768"],
            "ids_768_repeat": ids_768_repeat == ids_768,
        }
        return {
            "passed": all(matches.values()),
            "outputs": {"ids_128": ids_128, "ids_768": ids_768},
            "base_match": matches,
        }

    names = ["chat_single", "chat_multi"] * 4
    outputs = generate(llm, [cases[name] for name in names])
    expected_outputs = (
        [expected[f"{name}_{index}"] for index, name in enumerate(names)]
        if scenario == "graph"
        else [expected[name] for name in names]
    )
    first_matches = [
        output[:1] == target[:1] for output, target in zip(outputs, expected_outputs)
    ]
    full_matches = [output == target for output, target in zip(outputs, expected_outputs)]
    same_prompt = all(outputs[index] == outputs[0] for index in (2, 4, 6)) and all(
        outputs[index] == outputs[1] for index in (3, 5, 7)
    )
    return {
        "passed": all(full_matches) and (scenario == "graph" or same_prompt),
        "first_token_passed": all(first_matches),
        "outputs": {
            f"{name}_{i}": output for i, (name, output) in enumerate(zip(names, outputs))
        },
        "reference_first_token_match": first_matches,
        "reference_full_match": full_matches,
        "same_prompt_full_match": same_prompt,
    }


def run_perf(generate_fn, cases):
    generate_fn([cases["ids_32"]], 8)
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for input_len, output_len, batch_size in WORKLOADS:
        durations = []
        token_counts = []
        for repeat in range(2):
            prompts = []
            for index in range(batch_size):
                prompt = cases["ids_768"][:input_len]
                prompt = prompt[:-1] + [3 + (prompt[-1] + repeat + index) % 6397]
                prompts.append(prompt)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = generate_fn(prompts, output_len)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            token_counts.append(sum(len(output) for output in outputs))
        elapsed = sum(durations) / len(durations)
        name = f"in{input_len}_out{output_len}_bs{batch_size}"
        token_count = sum(token_counts) / len(token_counts)
        results[name] = {
            "seconds": elapsed,
            "requested_output_tokens": batch_size * output_len,
            "actual_output_tokens": token_count,
            "output_tokens_per_second": token_count / elapsed,
        }
    results["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    results["passed"] = True
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scenario",
        choices=("base", "batch", "chunk", "radix", "graph", "perf"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--runtime", choices=("minisgl", "native", "transformers"), default="minisgl"
    )
    args = parser.parse_args()

    cases = load_cases(args.baseline_dir)
    if args.scenario == "perf":
        if args.runtime == "minisgl":
            llm = create_llm(args.model, args.scenario)
            generate_fn = lambda prompts, max_tokens: generate(
                llm, prompts, max_tokens=max_tokens, ignore_eos=True
            )
        else:
            model = create_reference_model(args.runtime, args.model)
            generate_fn = lambda prompts, max_tokens: generate_reference(
                model, prompts, max_tokens
            )
        result = run_perf(generate_fn, cases)
    else:
        if args.runtime != "minisgl":
            parser.error("--runtime only applies to --scenario perf")
        llm = create_llm(args.model, args.scenario)
        result = run_correctness(
            llm, args.scenario, cases, args.reference, args.baseline_dir
        )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
