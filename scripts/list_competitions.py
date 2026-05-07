#!/usr/bin/env python3
"""List available Kaggle competitions."""
from dotenv import load_dotenv
import os
import sys

load_dotenv()
from kaggle import api

api.authenticate()
comps = api.competitions_list(page=1)

if hasattr(comps, 'competitions'):
    print(f"Found {len(comps.competitions)} competitions:\n")
    for c in comps.competitions[:10]:
        deadline = getattr(c, 'deadline', 'N/A')
        print(f"  {c.ref:40s} - {c.title[:50]}")
        print(f"    Deadline: {deadline}\n")
else:
    print(f"Response: {comps}")
    attrs = [a for a in dir(comps) if not a.startswith('_')]
    print(f"Attributes: {attrs}")
