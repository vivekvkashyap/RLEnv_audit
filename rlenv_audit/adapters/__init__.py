"""Adapters normalize an external environment format into an internal `EnvHandle`.

v0 ships exactly one adapter: `verifiers` (the Prime Intellect Hub format). The
package boundary exists so a second format could be added later without touching
the checks — but no such adapter is built now.
"""
