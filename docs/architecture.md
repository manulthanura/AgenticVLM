# Architecture

```
Video ─► ingestion ─► extract ─► state ─► events ─┬─► reasoning ─┬─► alerting ─► reporting
         sample 5fps   YOLO pose   per-frame       │  (only for   │   NORMAL /     timeline.txt
         + CLAHE       + ByteTrack candidates +     │  ambiguous   │   MONITOR /    summary.json
                       + features  smoothing        │  cases)      │   ALERT        events.json
                                                    │              │                alerts.json
                                                    └──────────────┘                agent_traces.json
                                          agent + Azure OpenAI VLM; skipped when not configured
```

`app/analysis` is the composition root: `RunAnalysis` runs the pipeline, `container.py` picks the concrete adapters,
`api/router.py` exposes it, `app/cli.py` is the command-line entry.

## Layers (per feature under `app/`)
- `domain/`: pure logic and ports (no cv2 / ultralytics / openai / fastapi).
- `application/`: use cases; depend on domain + ports only.
- `infrastructure/`: adapters (OpenCV, Ultralytics, Azure OpenAI, filesystem).
- `api/`: FastAPI routers.

## Data flow
1. `ingestion.OpenCVVideoSource` yields `SampledFrame`s at `SAMPLE_FPS`.
2. `extract.FeatureExtractor` (via the `PoseEstimator` port) yields `FrameFeatures` for the primary person.
3. `state.classify` gives a `StateCandidate` per frame; `StateMachine` smooths them into a `Timeline` of segments.
4. `events.BedEventDetector` finds bed exits and returns.
5. `reasoning.Reasoner` (optional) finds ambiguous cases, investigates each with the agent, resolves `UNKNOWN`
   segments, drops rejected exits and re-detects events.
6. `alerting.evaluate_alerts` applies the rules to the revised timeline, events and agent findings.
7. `reporting.render_report` writes the five output files to `outputs/<job_id>/`.

## Reasoning agent (`app/reasoning`)
Called only for cases the rules cannot settle (`find_cases`): lying outside the bed, short or weak bed exits, out of
view, a second person, long UNKNOWN. Per case, at most `AGENT_MAX_STEPS` (4) tool calls:

| Step | Tool | Uses LLM |
|---|---|---|
| 1 | `get_state_history` (segments around the case) | no |
| 2 | `get_track_info` (track, bed overlap, other people) | no |
| 3 | `ask_vlm`: 4 keyframes to the FAST deployment | yes |
| 4 | `ask_vlm`: wider, later window to the STRONG deployment, only if step 3 was below `AGENT_MIN_CONFIDENCE` | yes |

The "is the context sufficient?" decision is a plain rule (confidence vs threshold), so the loop is deterministic and
unit-testable; the VLM is a perception tool. The agent returns an `AgentFinding` and a full `AgentTrace`; `alerting`,
not the agent, decides ALERT.

Cost controls (`AzureVlm`): every call goes through a disk cache (`outputs/cache/`, key = hash of prompt version,
deployment, prompt and JPEG frames), images are ≤ 512 px at `detail: low`, 4 frames per call, and a hard cap of
`AGENT_MAX_CALLS_PER_VIDEO`. Replies must be JSON validated with Pydantic; invalid output is retried once, then becomes
an UNKNOWN answer. Prompts live in `app/reasoning/infrastructure/prompts/v1/*.txt`. Token usage is summed in
`agent_traces.json`.

Bed accounting: in bed = lying + sitting on bed; everything else, including UNKNOWN, is out of bed.
