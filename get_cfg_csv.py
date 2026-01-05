#!/usr/bin/env python3
import argparse, csv, sys, re
# from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3OmniMoeForConditionalGeneration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_id", help="e.g. openai/gpt-oss-20b or a local path")
    p.add_argument("--no_trust-remote", action="store_true", help="disable trust_remote_code")
    p.add_argument("--download", action="store_true", help="download weights (for real dtypes)")
    p.add_argument("--cfg-out", default="<model>_config.json")
    p.add_argument("--csv-out", default="<model>_tensors.csv")
    args = p.parse_args()

    trust_remote = not args.no_trust_remote
    trust_remote = True
    print(trust_remote)
    
    for attr in ["cfg_out", "csv_out"]:
        filename = getattr(args, attr)
        if '<model>' in filename:
            mod_model_id = re.sub('/', '-', args.model_id)
            setattr(args, attr, re.sub('<model>', mod_model_id, filename))

    # 1) Config
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=trust_remote)
    with open(args.cfg_out, "w") as f:
        f.write(cfg.to_json_string())

    # 2) Model (meta vs full)
    # --- Try to use the specific model class first (Required for this model) ---
    try:
        ModelClass = Qwen3OmniMoeForConditionalGeneration
    except NameError:
        # Fall back to AutoModel if the specific class isn't available
        ModelClass = AutoModelForCausalLM
    
    if args.download:
        model = ModelClass.from_pretrained(
            args.model_id, trust_remote_code=trust_remote, torch_dtype=None, device_map="cpu"
        )
    else:
        try:
            from accelerate import init_empty_weights
        except ImportError:
            sys.exit("Please `pip install accelerate` or use --download to load real weights.")
        with init_empty_weights():
            # Crucially, we use from_pretrained with config=cfg and the ModelClass
            model = ModelClass.from_pretrained(
                args.model_id, trust_remote_code=trust_remote, config=cfg
            )
    
    # 3) Gather rows
    rows = []
    for name, p in model.named_parameters(recurse=True):
        rows.append(("param", name, tuple(p.shape), str(p.dtype)))
    for name, b in model.named_buffers(recurse=True):
        rows.append(("buffer", name, tuple(b.shape), str(b.dtype)))

    # 4) CSV + console dump
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "shape", "dtype"])
        w.writerows(rows)

    # for kind, name, shape, dtype in rows:
    #     print(f"{name} {shape} {dtype}")

if __name__ == "__main__":
    main()
