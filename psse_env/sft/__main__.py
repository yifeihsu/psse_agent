import sys

if len(sys.argv) > 1 and sys.argv[1] == "research-audit":
    from .research_processor_audit import main

    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "research-views":
    from .research_views import main

    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "research-smoke":
    from .research_smoke import main

    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "research-cache":
    from .research_cache import main

    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "research-train":
    from .research_cli import main

    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "research-bc0-eval":
    from .research_bc0_eval import main

    raise SystemExit(main(sys.argv[2:]))

from .cli import main

raise SystemExit(main())
