"""
test_rag — Phase 1 acceptance test (the milestone, as code).

Goal: feed ONE API's docs (start with GitHub) through the full pipeline
  fetch -> chunk -> embed -> retrieve -> extract
and assert we get back a valid ApiSchema with the expected auth method,
endpoint, params, and pagination style.

TODO (Phase 1): implement once the rag/ modules exist. When this passes,
Phase 1 is done.
"""
