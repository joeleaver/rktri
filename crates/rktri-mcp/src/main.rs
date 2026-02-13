//! rktri-mcp — MCP debug server for Claude Code
//!
//! Connects to a running rktri instance's debug server over TCP
//! and exposes debug tools via the MCP protocol.

use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::io::BufReader as StdBufReader;
use std::net::TcpStream;
use std::sync::Mutex;

/// TCP connection to the debug server (lazy-initialized)
static DEBUG_CONN: Mutex<Option<DebugConnection>> = Mutex::new(None);

struct DebugConnection {
    stream: TcpStream,
}

impl DebugConnection {
    fn connect() -> io::Result<Self> {
        let stream = TcpStream::connect("127.0.0.1:9742")?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
        Ok(Self { stream })
    }

    fn send_command(&mut self, cmd: &Value) -> Result<Value, String> {
        let mut json_str = serde_json::to_string(cmd).map_err(|e| e.to_string())?;
        json_str.push('\n');
        self.stream
            .write_all(json_str.as_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
        self.stream
            .flush()
            .map_err(|e| format!("Flush error: {}", e))?;

        // Read response line
        let mut reader = StdBufReader::new(&self.stream);
        let mut response_line = String::new();
        reader
            .read_line(&mut response_line)
            .map_err(|e| format!("Read error: {}", e))?;

        serde_json::from_str(&response_line).map_err(|e| format!("Parse error: {}", e))
    }
}

fn send_debug_cmd(cmd: Value) -> Result<Value, String> {
    let mut conn_guard = DEBUG_CONN.lock().map_err(|e| e.to_string())?;

    if conn_guard.is_none() {
        *conn_guard = Some(DebugConnection::connect().map_err(|e| {
            format!(
                "Cannot connect to rktri debug server on port 9742. Is rktri running? Error: {}",
                e
            )
        })?);
    }

    let result = conn_guard.as_mut().unwrap().send_command(&cmd);

    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            // Reconnect and retry once
            log::warn!("Connection error, reconnecting: {}", e);
            *conn_guard = Some(DebugConnection::connect().map_err(|e| {
                format!("Reconnect failed: {}", e)
            })?);
            conn_guard.as_mut().unwrap().send_command(&cmd)
        }
    }
}

/// Tool definitions for MCP
fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "camera_move_to",
            "description": "Move the camera to an absolute world position",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "X position" },
                    "y": { "type": "number", "description": "Y position (up)" },
                    "z": { "type": "number", "description": "Z position" }
                },
                "required": ["x", "y", "z"]
            }
        }),
        json!({
            "name": "camera_move_relative",
            "description": "Move the camera relative to its current position",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dx": { "type": "number", "description": "Delta X" },
                    "dy": { "type": "number", "description": "Delta Y (up)" },
                    "dz": { "type": "number", "description": "Delta Z" }
                },
                "required": ["dx", "dy", "dz"]
            }
        }),
        json!({
            "name": "camera_look_at",
            "description": "Point the camera at a world position",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Target X" },
                    "y": { "type": "number", "description": "Target Y" },
                    "z": { "type": "number", "description": "Target Z" }
                },
                "required": ["x", "y", "z"]
            }
        }),
        json!({
            "name": "camera_get_state",
            "description": "Get current camera position, direction, and FOV",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "send_key",
            "description": "Send a keyboard key press to the application. Valid keys: W, A, S, D, T, F1-F5, Space, ShiftLeft, Escape, Tab",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "Key name (e.g. 'T', 'F1', 'W')" }
                },
                "required": ["key"]
            }
        }),
        json!({
            "name": "take_screenshot",
            "description": "Capture the current frame as a PNG image. Returns base64-encoded PNG data. The image can be viewed by reading the saved file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Optional file path to save the PNG to disk (e.g. '/tmp/screenshot.png')"
                    }
                }
            }
        }),
        json!({
            "name": "get_world_info",
            "description": "Get information about the loaded world: chunk count, extent, time of day, camera position",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "get_render_params",
            "description": "Get current rendering parameters including god rays, shadow, time of day, and debug mode settings",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "set_godrays_params",
            "description": "Adjust god rays (volumetric light scattering) parameters. Only specify the parameters you want to change.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "density": { "type": "number", "description": "Overall intensity multiplier (default 0.5)" },
                    "decay": { "type": "number", "description": "Per-step falloff, 0.9-0.99 (default 0.97)" },
                    "exposure": { "type": "number", "description": "Final brightness multiplier (default 0.01)" },
                    "weight": { "type": "number", "description": "Initial sample weight (default 1.0)" },
                    "num_samples": { "type": "integer", "description": "Ray march steps, 16-128 (default 64)" }
                }
            }
        }),
        json!({
            "name": "set_time_of_day",
            "description": "Set the time of day (affects sun position and lighting). Range: 0.0-24.0 where 12.0 is noon.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hour": { "type": "number", "description": "Hour (0.0-24.0)" }
                },
                "required": ["hour"]
            }
        }),
        json!({
            "name": "get_fps_stats",
            "description": "Get FPS statistics with 1s, 5s, and 15s rolling averages including min/max values",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "set_weather",
            "description": "Set the weather preset (clear, partlycloudy, overcast, foggy, rain, snow, storm)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "preset": { "type": "string", "description": "Weather preset name" }
                },
                "required": ["preset"]
            }
        }),
        json!({
            "name": "set_fog",
            "description": "Adjust fog parameters. Only specify parameters you want to change.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "enabled": { "type": "boolean", "description": "Enable/disable fog" },
                    "density": { "type": "number", "description": "Distance fog density" },
                    "height_falloff": { "type": "number", "description": "Height fog falloff rate" },
                    "height_base": { "type": "number", "description": "Height fog base altitude" },
                    "inscattering": { "type": "number", "description": "Inscattering intensity" }
                }
            }
        }),
        json!({
            "name": "set_wind",
            "description": "Adjust wind parameters. Only specify parameters you want to change.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "direction_x": { "type": "number", "description": "Wind direction X component" },
                    "direction_z": { "type": "number", "description": "Wind direction Z component" },
                    "speed": { "type": "number", "description": "Wind base speed" },
                    "gust_strength": { "type": "number", "description": "Wind gust strength multiplier" }
                }
            }
        }),
        json!({
            "name": "get_atmosphere_state",
            "description": "Get current atmosphere state including sun, fog, weather, wind",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "set_grass_params",
            "description": "Adjust procedural grass parameters. Only specify parameters you want to change.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "enabled": { "type": "boolean", "description": "Enable/disable procedural grass" },
                    "density": { "type": "number", "description": "Blade placement probability 0.0-1.0 (default 0.8)" },
                    "blade_height_min": { "type": "number", "description": "Minimum blade height in meters (default 0.4)" },
                    "blade_height_max": { "type": "number", "description": "Maximum blade height in meters (default 1.0)" },
                    "blade_width": { "type": "number", "description": "Blade width in meters (default 0.025)" },
                    "sway_amount": { "type": "number", "description": "Wind sway influence 0.0-2.0 (default 0.5)" },
                    "max_distance": { "type": "number", "description": "Render cutoff distance in meters (default 30.0)" },
                    "coverage_scale": { "type": "number", "description": "Coverage noise scale in meters (default 5.0)" },
                    "coverage_amount": { "type": "number", "description": "Coverage noise influence 0.0-1.0 (default 0.6)" }
                }
            }
        }),
        json!({
            "name": "get_grass_state",
            "description": "Get current procedural grass parameters",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "set_dlss_mode",
            "description": "Set DLSS quality mode (off, auto, quality, balanced, performance, ultraperformance, dlaa)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "DLSS quality mode"
                    }
                },
                "required": ["mode"]
            }
        }),
        json!({
            "name": "get_dlss_state",
            "description": "Get current DLSS upscaling state (enabled, quality mode, resolution)",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "raycast_down",
            "description": "Raycast straight down to find terrain height at (x, z). Returns the estimated terrain surface Y coordinate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "X position" },
                    "z": { "type": "number", "description": "Z position" }
                },
                "required": ["x", "z"]
            }
        }),
        json!({
            "name": "ping",
            "description": "Check if the debug server is connected and responsive",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
    ]
}

/// Handle a tool call
fn handle_tool_call(name: &str, args: &Value) -> Value {
    let cmd = match name {
        "camera_move_to" => json!({
            "cmd": "CameraMoveTo",
            "params": {
                "x": args["x"].as_f64().unwrap_or(0.0),
                "y": args["y"].as_f64().unwrap_or(0.0),
                "z": args["z"].as_f64().unwrap_or(0.0)
            }
        }),
        "camera_move_relative" => json!({
            "cmd": "CameraMoveRelative",
            "params": {
                "dx": args["dx"].as_f64().unwrap_or(0.0),
                "dy": args["dy"].as_f64().unwrap_or(0.0),
                "dz": args["dz"].as_f64().unwrap_or(0.0)
            }
        }),
        "camera_look_at" => json!({
            "cmd": "CameraLookAt",
            "params": {
                "x": args["x"].as_f64().unwrap_or(0.0),
                "y": args["y"].as_f64().unwrap_or(0.0),
                "z": args["z"].as_f64().unwrap_or(0.0)
            }
        }),
        "camera_get_state" => json!({
            "cmd": "CameraGetState"
        }),
        "send_key" => json!({
            "cmd": "SendKey",
            "params": { "key": args["key"].as_str().unwrap_or("") }
        }),
        "take_screenshot" => json!({
            "cmd": "TakeScreenshot"
        }),
        "get_world_info" => json!({
            "cmd": "GetWorldInfo"
        }),
        "get_render_params" => json!({
            "cmd": "GetRenderParams"
        }),
        "set_godrays_params" => {
            let mut params = serde_json::Map::new();
            if let Some(v) = args.get("density") {
                params.insert("density".into(), v.clone());
            }
            if let Some(v) = args.get("decay") {
                params.insert("decay".into(), v.clone());
            }
            if let Some(v) = args.get("exposure") {
                params.insert("exposure".into(), v.clone());
            }
            if let Some(v) = args.get("weight") {
                params.insert("weight".into(), v.clone());
            }
            if let Some(v) = args.get("num_samples") {
                params.insert("num_samples".into(), v.clone());
            }
            json!({
                "cmd": "SetGodRaysParams",
                "params": params
            })
        }
        "set_time_of_day" => json!({
            "cmd": "SetTimeOfDay",
            "params": { "hour": args["hour"].as_f64().unwrap_or(12.0) }
        }),
        "get_fps_stats" => json!({
            "cmd": "GetFpsStats"
        }),
        "set_weather" => json!({
            "cmd": "SetWeather",
            "params": { "preset": args["preset"].as_str().unwrap_or("clear") }
        }),
        "set_fog" => {
            let mut params = serde_json::Map::new();
            if let Some(v) = args.get("enabled") {
                params.insert("enabled".into(), v.clone());
            }
            if let Some(v) = args.get("density") {
                params.insert("density".into(), v.clone());
            }
            if let Some(v) = args.get("height_falloff") {
                params.insert("height_falloff".into(), v.clone());
            }
            if let Some(v) = args.get("height_base") {
                params.insert("height_base".into(), v.clone());
            }
            if let Some(v) = args.get("inscattering") {
                params.insert("inscattering".into(), v.clone());
            }
            json!({
                "cmd": "SetFog",
                "params": params
            })
        }
        "set_wind" => {
            let mut params = serde_json::Map::new();
            if let Some(v) = args.get("direction_x") {
                params.insert("direction_x".into(), v.clone());
            }
            if let Some(v) = args.get("direction_z") {
                params.insert("direction_z".into(), v.clone());
            }
            if let Some(v) = args.get("speed") {
                params.insert("speed".into(), v.clone());
            }
            if let Some(v) = args.get("gust_strength") {
                params.insert("gust_strength".into(), v.clone());
            }
            json!({
                "cmd": "SetWind",
                "params": params
            })
        }
        "get_atmosphere_state" => json!({
            "cmd": "GetAtmosphereState"
        }),
        "set_grass_params" => {
            let mut params = serde_json::Map::new();
            if let Some(v) = args.get("enabled") {
                params.insert("enabled".into(), v.clone());
            }
            if let Some(v) = args.get("density") {
                params.insert("density".into(), v.clone());
            }
            if let Some(v) = args.get("blade_height_min") {
                params.insert("blade_height_min".into(), v.clone());
            }
            if let Some(v) = args.get("blade_height_max") {
                params.insert("blade_height_max".into(), v.clone());
            }
            if let Some(v) = args.get("blade_width") {
                params.insert("blade_width".into(), v.clone());
            }
            if let Some(v) = args.get("sway_amount") {
                params.insert("sway_amount".into(), v.clone());
            }
            if let Some(v) = args.get("max_distance") {
                params.insert("max_distance".into(), v.clone());
            }
            if let Some(v) = args.get("coverage_scale") {
                params.insert("coverage_scale".into(), v.clone());
            }
            if let Some(v) = args.get("coverage_amount") {
                params.insert("coverage_amount".into(), v.clone());
            }
            json!({
                "cmd": "SetGrassParams",
                "params": params
            })
        }
        "get_grass_state" => json!({
            "cmd": "GetGrassState"
        }),
        "set_dlss_mode" => json!({
            "cmd": "SetDlssMode",
            "params": {
                "mode": args["mode"].as_str().unwrap_or("auto")
            }
        }),
        "get_dlss_state" => json!({
            "cmd": "GetDlssState"
        }),
        "raycast_down" => json!({
            "cmd": "RaycastDown",
            "params": {
                "x": args["x"].as_f64().unwrap_or(0.0) as f32,
                "z": args["z"].as_f64().unwrap_or(0.0) as f32
            }
        }),
        "ping" => json!({ "cmd": "Ping" }),
        _ => {
            return json!({
                "content": [{"type": "text", "text": format!("Unknown tool: {}", name)}],
                "isError": true
            });
        }
    };

    match send_debug_cmd(cmd) {
        Ok(response) => {
            // Check if screenshot and save_path provided
            if name == "take_screenshot" {
                if let Some(save_path) = args.get("save_path").and_then(|p| p.as_str()) {
                    if let Some(data) = response
                        .get("data")
                        .and_then(|d| d.get("png_base64"))
                        .and_then(|b| b.as_str())
                    {
                        use base64::Engine as _;
                        match base64::engine::general_purpose::STANDARD.decode(data) {
                            Ok(bytes) => {
                                if let Err(e) = std::fs::write(save_path, &bytes) {
                                    return json!({
                                        "content": [{"type": "text", "text": format!("Screenshot taken but save failed: {}", e)}],
                                        "isError": true
                                    });
                                }
                                let w = response
                                    .get("data")
                                    .and_then(|d| d.get("width"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let h = response
                                    .get("data")
                                    .and_then(|d| d.get("height"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                return json!({
                                    "content": [{"type": "text", "text": format!("Screenshot saved to {} ({}x{})", save_path, w, h)}]
                                });
                            }
                            Err(e) => {
                                return json!({
                                    "content": [{"type": "text", "text": format!("Failed to decode screenshot: {}", e)}],
                                    "isError": true
                                });
                            }
                        }
                    }
                }
            }

            let text =
                serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".into());
            json!({
                "content": [{"type": "text", "text": text}]
            })
        }
        Err(e) => {
            json!({
                "content": [{"type": "text", "text": format!("Debug server error: {}", e)}],
                "isError": true
            })
        }
    }
}

/// Process a JSON-RPC request and return a response
fn process_request(request: &Value) -> Option<Value> {
    let id = request.get("id").cloned();
    let method = request
        .get("method")
        .and_then(|m| m.as_str())
        .unwrap_or("");

    let result = match method {
        "initialize" => {
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "rktri-mcp",
                    "version": "0.1.0"
                }
            })
        }
        "notifications/initialized" => {
            // Notification — no response needed
            return None;
        }
        "tools/list" => {
            json!({
                "tools": tool_definitions()
            })
        }
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let tool_name = params
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let tool_args = params.get("arguments").cloned().unwrap_or(json!({}));
            handle_tool_call(tool_name, &tool_args)
        }
        "ping" => {
            json!({})
        }
        _ => {
            // Unknown method — return error
            return Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {}", method)
                }
            }));
        }
    };

    Some(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }))
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .init();

    log::info!("rktri-mcp server starting");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                log::error!("stdin read error: {}", e);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Invalid JSON: {}", e);
                let error_resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                let _ = writeln!(stdout_lock, "{}", error_resp);
                let _ = stdout_lock.flush();
                continue;
            }
        };

        if let Some(response) = process_request(&request) {
            if let Err(e) = writeln!(stdout_lock, "{}", response) {
                log::error!("stdout write error: {}", e);
                break;
            }
            if let Err(e) = stdout_lock.flush() {
                log::error!("stdout flush error: {}", e);
                break;
            }
        }
    }

    log::info!("rktri-mcp server shutting down");
}
