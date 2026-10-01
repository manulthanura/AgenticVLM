from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All thresholds live here (overridable via .env). Domain code receives values, never reads env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI (deployment names come from env, never hard-coded)
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = ""
    azure_openai_deployment_fast: str = ""
    azure_openai_deployment_strong: str = ""

    # Folders (data/videos, data/rois, outputs/<job_id>/)
    data_dir: str = "data"
    outputs_dir: str = "outputs"

    # Perception / ingestion
    pose_model: str = (
        "yolo26s-pose.pt"  # nano missed a prone person in half the frames; small found 86 %
    )
    sample_fps: float = 5
    detection_conf_min: float = 0.25  # Ultralytics default; below this boxes are mostly noise
    keypoint_conf_min: float = 0.3  # a keypoint under this is treated as not visible
    edge_margin_px: float = 30  # hips this close to the bed border count as "on the edge"
    relock_max_dist_bbox_heights: float = (
        1.5  # max jump when re-adopting the person after an ID switch
    )

    # State machine
    min_state_dwell_sec: float = 2  # ignore flicker shorter than this
    dwell_share_min: float = (
        0.7  # share of the dwell window a new state must fill (tolerates flicker)
    )
    state_confidence_min: float = 0.5  # weaker candidates count as UNKNOWN, not a guess
    visible_ratio_min: float = (
        0.4  # blankets hide legs (~0.6 visible); below 0.4 the pose is unreliable
    )
    lying_torso_angle_deg: float = 60  # torso this far from vertical = horizontal posture
    seated_knee_max_deg: float = 145  # seated knees ~90-130 deg, standing ~160-180: split the gap
    reclined_torso_angle_deg: float = (
        35  # propped on a pillow reads 38-52 deg; only trusted with hips on the bed
    )
    speed_window_sec: float = (
        1.0  # single 0.2 s steps are too jittery to tell walking from standing
    )
    speed_jump_max: float = (
        1.5  # bbox heights/s in one step: faster is a track jump (ID switch), not a person
    )
    lying_bbox_aspect_min: float = (
        1.0  # box at least as wide as tall = lying, whichever way the body faces the camera
    )
    walking_speed_min: float = 0.3  # bbox heights/s; sitting-up sway stays below, a stride is ~0.5+

    # Bed events
    bed_exit_confirm_sec: float = 5  # standing briefly then sitting back is not an exit
    bed_return_confirm_sec: float = 5
    event_match_tolerance_sec: float = 5

    # Evaluation
    evaluation_sample_sec: float = 1  # frame accuracy is measured once per second
    failure_cases_max: int = 5  # the brief wants at least 3 documented failures

    # Alert rules
    edge_sitting_monitor_sec: float = 180
    unknown_monitor_sec: float = 30
    out_of_bed_alert_sec: float = 600
    out_of_view_alert_sec: float = 300

    # Agent
    agent_max_steps: int = 4
    agent_max_calls_per_video: int = 30
    unknown_escalate_sec: float = 10  # UNKNOWN longer than this is sent to the agent
    agent_min_confidence: float = 0.6  # below this a finding is weak: escalate once, then MONITOR
    agent_context_sec: float = 10  # look this far around a case; also the widest frame window
    agent_frames_per_call: int = 4  # 3-4 frames give the VLM motion context at low cost
    agent_image_max_side: int = 512  # downscale before sending; detail is not needed for posture
    multiple_persons_min_sec: float = 3  # a passer-by for a moment is not worth a VLM call


def get_settings() -> Settings:
    return Settings()
