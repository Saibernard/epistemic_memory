#!/usr/bin/env python3
"""
Memory Layer MCP Server — back-compat shim.

The server now lives inside the package (memory_layer/mcp.py) so it works
identically from a source checkout and a pip install.  This file remains
so existing Claude Code / Cursor configs that point at
`python mcp_server.py` keep working.

Preferred invocation:  memory-layer mcp
"""

import os
import sys

# Running from a source checkout: make the adjacent package importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_layer.mcp import main

if __name__ == "__main__":
    main()
