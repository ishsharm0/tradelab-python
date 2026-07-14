# Security policy

## Supported versions

Security fixes are provided for the latest released version of TradeLab for Python.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for
`ishsharm0/tradelab-python`. Do not open a public issue for credential exposure,
order-routing defects, path traversal, or live-trading permission bypasses.

Include the affected version, a minimal reproduction, impact, and any suggested
mitigation. Please remove API keys, account identifiers, order IDs, positions, and
other financial information from reports and logs.

## Live-trading boundary

Paper mode is the default. Live session creation requires all of the following:

1. `TRADELAB_ALLOW_LIVE=true` in the process environment.
2. Explicit `confirm_live=True` authorization for that session.
3. A connected, credentialed, non-paper broker adapter.

Treat broker credentials as secrets. Supply them at runtime, never commit them,
and use restricted or testnet accounts while integrating an adapter. TradeLab does
not guarantee profitability and is not a substitute for broker-side risk controls.
