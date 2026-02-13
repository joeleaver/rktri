//! Logging initialization and utilities

/// Initialize the logging system
///
/// Uses env_logger with default filter level of `info`.
/// Override with RUST_LOG environment variable.
///
/// # Example
/// ```
/// rktri::core::logging::init();
/// log::info!("Engine started");
/// ```
pub fn init() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();
}
