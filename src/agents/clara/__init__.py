"""Email agent (code name ``clara``): Gmail → triage → task → reply draft.

The graph is tool-free; every side effect (Gmail, backend) lives in
``runner.py`` and runs only after an explicit human decision in the CRM.
"""
