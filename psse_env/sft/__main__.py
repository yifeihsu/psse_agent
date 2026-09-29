import sys

if len(sys.argv) > 1 and sys.argv[1] == "research-train":
    from .research_cli import main

    raise SystemExit(main(sys.argv[2:]))

raise SystemExit("usage: python -m psse_env.sft research-train [options]")
