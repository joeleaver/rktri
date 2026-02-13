//! TCP debug server

use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::protocol::{DebugCommand, DebugResponse};

/// Trait that the application implements to handle debug commands
pub trait DebugHandler: Send + Sync + 'static {
    fn handle_command(&mut self, cmd: DebugCommand) -> DebugResponse;
}

/// Debug server handle - keep this alive to keep the server running
pub struct DebugServer {
    _handle: tokio::task::JoinHandle<()>,
}

impl DebugServer {
    /// Start the debug server on the given port.
    /// The handler is called for each incoming command.
    /// Returns immediately -- server runs in background.
    pub fn start(handler: Arc<Mutex<dyn DebugHandler>>, port: u16) -> Self {
        let handle = tokio::spawn(async move {
            let addr = format!("127.0.0.1:{}", port);
            let listener = match TcpListener::bind(&addr).await {
                Ok(l) => {
                    log::info!("Debug server listening on {}", addr);
                    l
                }
                Err(e) => {
                    log::error!("Failed to bind debug server on {}: {}", addr, e);
                    return;
                }
            };

            loop {
                match listener.accept().await {
                    Ok((stream, peer)) => {
                        log::info!("Debug client connected from {}", peer);
                        let handler = handler.clone();
                        tokio::spawn(async move {
                            handle_connection(stream, handler).await;
                            log::info!("Debug client disconnected: {}", peer);
                        });
                    }
                    Err(e) => {
                        log::error!("Debug server accept error: {}", e);
                    }
                }
            }
        });

        Self { _handle: handle }
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    handler: Arc<Mutex<dyn DebugHandler>>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // Connection closed
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let response = match serde_json::from_str::<DebugCommand>(trimmed) {
                    Ok(cmd) => {
                        log::debug!("Debug command: {:?}", cmd);
                        let mut h = handler.lock().await;
                        h.handle_command(cmd)
                    }
                    Err(e) => DebugResponse::error(format!("Invalid command JSON: {}", e)),
                };

                let mut resp_json =
                    serde_json::to_string(&response).unwrap_or_else(|e| {
                        format!(
                            "{{\"status\":\"error\",\"message\":\"Serialize error: {}\"}}",
                            e
                        )
                    });
                resp_json.push('\n');

                if let Err(e) = writer.write_all(resp_json.as_bytes()).await {
                    log::error!("Debug server write error: {}", e);
                    break;
                }
                if let Err(e) = writer.flush().await {
                    log::error!("Debug server flush error: {}", e);
                    break;
                }
            }
            Err(e) => {
                log::error!("Debug server read error: {}", e);
                break;
            }
        }
    }
}
