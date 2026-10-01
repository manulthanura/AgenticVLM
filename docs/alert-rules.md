# Alert rules

Deterministic rules in `app/alerting/domain/rules.py`. Each fired rule records its name, the time it fired and evidence
(start/end/duration, limit, mean confidence, states). The overall decision is the most severe one (`NORMAL` if nothing fired).
Limits are configurable in `.env` (see `app/shared/config.py`).

| Rule | Decision | Fires when | Why |
|---|---|---|---|
| `bed_exit` | MONITOR | a confirmed bed exit | Leaving bed is the moment falls happen, especially at night. Not an alert on its own |
| `return_to_bed` | NORMAL | a confirmed return | Reassuring closure of an exit |
| `lying_outside_bed` | ALERT | an UNKNOWN segment with reason `lying_outside_bed` (immediately, after the 2 s state dwell) | Horizontal posture away from the bed is a possible fall; waiting only delays help |
| `out_of_bed_too_long` | ALERT | one out-of-bed stretch lasts > `OUT_OF_BED_ALERT_SEC` (600 s), fired at start + 600 s | Someone up for 10+ minutes may be unwell or lost. Needs at least one non-UNKNOWN state, so a camera blind spot alone does not count |
| `out_of_view_too_long` | ALERT | `OUT_OF_BED` (not visible) > `OUT_OF_VIEW_ALERT_SEC` (300 s) | We cannot see them and they did not come back |
| `edge_sitting_too_long` | MONITOR | `SITTING_ON_BED` with hips on the bed edge > `EDGE_SITTING_MONITOR_SEC` (180 s) | Long edge-sitting can mean difficulty standing up. Sitting inside the bed (reading) is not flagged |
| `unknown_too_long` | MONITOR | UNKNOWN > `UNKNOWN_MONITOR_SEC` (30 s), excluding `lying_outside_bed` | We cannot vouch for the person. The excluded case already has its own ALERT |
| `agent_low_confidence` | MONITOR | an agent finding has confidence < `AGENT_MIN_CONFIDENCE` (0.6), including "budget exhausted" and VLM errors | If even the agent could not settle a case, a human should look. It is never an ALERT on its own |

## How the agent interacts with the rules
The agent (`app/reasoning`) never decides `ALERT`. It only resolves ambiguity *before* the rules run:
- A confident finding (≥ 0.6) can relabel an `UNKNOWN` segment. A `lying_outside_bed` segment resolved to
  `LYING_IN_BED` (bed drawn slightly off, blanket) therefore no longer fires `lying_outside_bed`.
  If the agent says "on the floor" the state stays `UNKNOWN` and the ALERT stands.
- A bed exit the agent confidently judges to be "stood up and sat back" is removed from the events.
- Confident non-UNKNOWN states are never overridden.
- With no Azure settings the agent is skipped and only the rules above apply.

## Bed accounting
In bed = `LYING_IN_BED` + `SITTING_ON_BED`. Everything else, **including `UNKNOWN`**, counts as out of bed, matching the brief's example.

## Not implemented yet
- "Especially at night" for bed exits: needs a brightness/time-of-day signal; exits are MONITOR regardless.
