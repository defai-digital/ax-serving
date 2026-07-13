# Monitoring integration baseline

This directory contains a Grafana dashboard and Prometheus Operator alert
rules for the portable gateway. They are templates, not production evidence.

- Import `grafana-dashboard.json` into the environment's Grafana provisioning
  system.
- Apply `prometheus-rules.yaml` only when the Prometheus Operator CRDs exist.
- Configure the scraper to send `Authorization: Bearer <admin key>` to
  `/metrics`. Keep that credential in the monitoring system's secret store.
- Do not expose the bundled `/dashboard` directly. It is a compatibility UI
  whose authenticated browser requests require an admin reverse proxy. Use
  this Grafana dashboard for production monitoring.
- Replace provisional failure and retry thresholds with values derived from
  retained load and soak evidence before paging operators.
- Verify every `runbook_url` resolves in the published documentation system.

The dashboard and rules use only bounded aggregate labels. Request IDs,
worker IDs, model paths, prompts, outputs, and credentials remain outside
Prometheus labels.
