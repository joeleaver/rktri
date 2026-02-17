//! Debug protocol - JSON command/response definitions

use serde::{Deserialize, Serialize};

/// Commands sent from MCP server to debug server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "params")]
pub enum DebugCommand {
    /// Move camera to absolute position
    CameraMoveTo { x: f32, y: f32, z: f32 },
    /// Move camera relative to current position
    CameraMoveRelative { dx: f32, dy: f32, dz: f32 },
    /// Look at a target point
    CameraLookAt { x: f32, y: f32, z: f32 },
    /// Get current camera state
    CameraGetState,
    /// Send a key press event
    SendKey { key: String },
    /// Take a screenshot (returns base64 PNG)
    TakeScreenshot,
    /// Get world/scene information
    GetWorldInfo,
    /// Get current render parameters
    GetRenderParams,
    /// Set god rays parameters
    SetGodRaysParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        density: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decay: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exposure: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        weight: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        num_samples: Option<u32>,
    },
    /// Set time of day (0.0-24.0)
    SetTimeOfDay { hour: f32 },
    /// Get FPS statistics (1s/5s/15s averages with min/max)
    GetFpsStats,
    /// Get GPU profiling statistics (per-pass timing)
    GetProfileStats,
    /// Set render scale (0.25-2.0)
    SetRenderScale { scale: f32 },
    /// Set weather preset
    SetWeather { preset: String },
    /// Set fog parameters (only specified fields are updated)
    SetFog {
        #[serde(skip_serializing_if = "Option::is_none")]
        enabled: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        density: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        height_falloff: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        height_base: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        inscattering: Option<f32>,
    },
    /// Set wind parameters
    SetWind {
        #[serde(skip_serializing_if = "Option::is_none")]
        direction_x: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        direction_z: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        speed: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        gust_strength: Option<f32>,
    },
    /// Get full atmosphere state
    GetAtmosphereState,
    /// Set grass parameters (only specified fields are updated)
    SetGrassParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        enabled: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        density: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blade_height_min: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blade_height_max: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blade_width: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sway_amount: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_distance: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        coverage_scale: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        coverage_amount: Option<f32>,
    },
    /// Get current grass state
    GetGrassState,
    /// Set DLSS quality mode (off, auto, quality, balanced, performance, ultraperformance, dlaa)
    SetDlssMode {
        mode: String,
    },
    /// Get current DLSS state
    GetDlssState,
    /// Raycast straight down from (x, high_y, z) to find terrain height
    RaycastDown { x: f32, z: f32 },
    /// Get world metadata from manifest.json
    GetWorldMetadata,
    /// Get info for a specific chunk (if loaded)
    GetChunkInfo { x: i32, y: i32, z: i32 },
    /// Ping (health check)
    Ping,
}

/// Responses from debug server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum DebugResponse {
    #[serde(rename = "ok")]
    Ok { data: ResponseData },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Response data variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseData {
    None,
    Pong { message: String },
    CameraState {
        position: [f32; 3],
        forward: [f32; 3],
        fov_degrees: f32,
    },
    Screenshot {
        width: u32,
        height: u32,
        png_base64: String,
    },
    WorldInfo {
        chunk_count: u32,
        world_extent: f32,
        time_of_day: f32,
        camera_position: [f32; 3],
    },
    RenderParams {
        time_of_day: f32,
        debug_mode: u32,
        godrays: GodRaysInfo,
        shadow: ShadowInfo,
    },
    ParamsUpdated { description: String },
    FpsStats {
        current_fps: f32,
        frame_count: u64,
        one_sec: FpsWindowInfo,
        five_sec: FpsWindowInfo,
        fifteen_sec: FpsWindowInfo,
    },
    ProfileStats {
        enabled: bool,
        svo_trace_ms: f32,
        shadow_ms: f32,
        godrays_ms: f32,
        lighting_ms: f32,
        display_ms: f32,
        total_gpu_ms: f32,
    },
    AtmosphereInfo {
        time_of_day: f32,
        sun_direction: [f32; 3],
        sun_color: [f32; 3],
        sun_intensity: f32,
        ambient_color: [f32; 3],
        fog_density: f32,
        fog_enabled: bool,
        weather_preset: String,
        wind_direction: [f32; 3],
        wind_speed: f32,
        cloud_coverage: f32,
    },
    GrassInfo {
        enabled: bool,
        density: f32,
        blade_height_min: f32,
        blade_height_max: f32,
        blade_width: f32,
        sway_amount: f32,
        max_distance: f32,
        fade_start: f32,
        coverage_scale: f32,
        coverage_amount: f32,
    },
    DlssState {
        enabled: bool,
        supported: bool,
        quality_mode: String,
        render_width: u32,
        render_height: u32,
        upscaled_width: u32,
        upscaled_height: u32,
    },
    TerrainHeight {
        x: f32,
        z: f32,
        height: f32,
        hit: bool,
    },
    WorldMetadata {
        name: String,
        version: u32,
        seed: u32,
        size: f32,
        chunk_size: u32,
        terrain_params: TerrainParamsMeta,
        layers: Vec<LayerMeta>,
    },
    ChunkInfo {
        x: i32,
        y: i32,
        z: i32,
        loaded: bool,
        node_count: u32,
        brick_count: u32,
        world_min: Option<[f32; 3]>,
        has_grass: bool,
        grass_nodes: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodRaysInfo {
    pub density: f32,
    pub decay: f32,
    pub exposure: f32,
    pub weight: f32,
    pub num_samples: u32,
    pub sun_screen_pos: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowInfo {
    pub soft_shadow_samples: u32,
    pub soft_shadow_angle: f32,
    pub shadow_bias: f32,
    pub leaf_opacity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpsWindowInfo {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainParamsMeta {
    pub scale: f32,
    pub height_scale: f32,
    pub octaves: u32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub sea_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMeta {
    pub name: String,
    pub id: u32,
    pub directory: String,
    pub chunk_count: u32,
    pub total_bytes: u64,
}

impl DebugResponse {
    pub fn ok(data: ResponseData) -> Self {
        Self::Ok { data }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error {
            message: msg.into(),
        }
    }

    pub fn pong() -> Self {
        Self::ok(ResponseData::Pong {
            message: "pong".into(),
        })
    }

    pub fn none() -> Self {
        Self::ok(ResponseData::None)
    }
}
