//! Voxel edit system with delta-based persistence.
//!
//! Edits are stored as deltas over the base voxel data,
//! enabling undo/redo and optional persistence.

pub mod delta;
pub mod overlay;
pub mod invalidator;
pub mod brick_updater;
pub mod log;

pub use delta::{EditDelta, EditOp};
pub use overlay::EditOverlay;
pub use invalidator::ChunkInvalidator;
pub use brick_updater::BrickUpdater;
pub use log::EditLog;
