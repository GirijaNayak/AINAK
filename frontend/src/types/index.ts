export interface SystemStatus {
  app_name: string;
  version: string;
  python_version: string;
  torch_version: string;
  device: string;
  cuda_available: boolean;
  platform: string;
  automated_tests: {
    passed: number;
    failed: number;
    skipped: number;
    total: number;
    status: string;
  };
  system_resources?: {
    cpu_percent?: number;
    memory_used_gb?: number;
    memory_total_gb?: number;
    storage_free_gb?: number;
  };
}

export interface ModelDetail {
  available: boolean;
  model_name: string;
  checkpoint_path: string;
  checkpoint_sha256?: string;
  file_size_mb?: number;
  architecture?: string;
  latency_ms?: number;
  parameters_count?: number | string;
  classes?: string[] | number;
}

export interface ModelsStatus {
  vision: ModelDetail;
  audio: ModelDetail;
  gesture: ModelDetail;
  fusion: ModelDetail;
}

export interface SummaryMetrics {
  vision: {
    dataset: string;
    map_50: number;
    map_50_95: number;
    precision: number;
    recall: number;
    latency_ms: number;
  };
  audio: {
    dataset: string;
    accuracy: number;
    macro_f1: number;
    latency_ms: number;
    noise_robustness?: Record<string, { accuracy: number; macro_f1: number }>;
  };
  gesture: {
    dataset: string;
    accuracy: number;
    macro_f1: number;
    latency_ms: number;
    samples_count: number;
  };
  fusion: {
    architecture: string;
    priority_mae: number;
    best_val_loss: number;
    latency_ms: number;
  };
}

export interface ReportsSummary {
  summary_metrics: SummaryMetrics;
  vision?: any;
  audio?: any;
  gesture?: any;
  fusion?: any;
}

export interface SessionInfo {
  session_id: string;
  has_video: boolean;
  has_audio: boolean;
  has_trace: boolean;
  video_duration?: number;
  audio_duration?: number;
}

export interface VisionDetection {
  label: string;
  confidence: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  track_id: number;
  depth: number;
  hazard_score: number;
  is_critical: boolean;
}

export interface VisionInferenceResult {
  detections: VisionDetection[];
  detected_objects_count: number;
  max_hazard_score: number;
  overall_hazard_level: 'none' | 'low' | 'medium' | 'high' | 'imminent';
  latency_ms: number;
  timestamp?: number;
  feature_vector?: number[];
}

export interface AudioClassProb {
  class_name: string;
  probability: number;
}

export interface AudioInferenceResult {
  predicted_class: string;
  probability: number;
  top_classes: AudioClassProb[];
  is_silent: boolean;
  rms_energy: number;
  latency_ms: number;
  timestamp?: number;
  feature_vector?: number[];
}

export interface GestureInferenceResult {
  gesture_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  landmarks: Array<[number, number, number, number]>; // 33 points of [x, y, z, vis]
  sequence_buffer_len: number;
  latency_ms: number;
  timestamp?: number;
  feature_vector?: number[];
}

export interface FusionInferenceResult {
  raw_output: {
    priority_scalar: number;
    confidence_scalar: number;
    alert_logits: number[];
  };
  interpreted_prediction: {
    alert_type: string;
    priority_pct: number;
    confidence_pct: number;
    alert_probabilities?: Record<string, number>;
  };
  decision_engine_output: {
    dispatched: boolean;
    alert_rank: number;
    alert_type: string;
    cooldown_active: boolean;
    speech_interrupted: boolean;
    spoken_text?: string;
  };
  contributing_modalities: string[];
  latency_ms: number;
}

export interface ReplayTimelineStep {
  step_index: number;
  total_steps: number;
  timestamp_seconds: number;
  contributing_modalities: string[];
  modality_validity: {
    vision: boolean;
    audio: boolean;
    gesture: boolean;
  };
  transformer_used: boolean;
  fusion_output?: {
    priority_score: number;
    confidence: number;
    alert_type: string;
    alert_probabilities: Record<string, number>;
  };
  decision_engine_output?: {
    dispatched: boolean;
    alert_rank: number;
    alert_type: string;
    cooldown_active: boolean;
    spoken_alert?: string;
  };
  source_predictions?: {
    vision?: {
      objects: Array<{ label: string; confidence: number; track_id: number; hazard_score: number }>;
    };
    audio?: {
      predicted_class: string;
      probability: number;
      is_silent: boolean;
    };
    gesture?: {
      gesture_class: string;
      confidence: number;
    };
  };
  stage_latency_ms?: {
    vision?: number;
    audio?: number;
    gesture?: number;
    fusion?: number;
  };
}
