//! Debug tools for rktri - TCP debug server for remote inspection and control
//!
//! Start the debug server in your app:
//! ```ignore
//! let handler = Arc::new(Mutex::new(MyHandler::new()));
//! let _server = DebugServer::start(handler, 9742);
//! ```

pub mod protocol;
pub mod server;

pub use protocol::*;
pub use server::{DebugHandler, DebugServer};

/// Default debug server port
pub const DEFAULT_PORT: u16 = 9742;
