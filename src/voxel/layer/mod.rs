//! Multi-layer voxel system for independent streaming and rendering.
//!
//! Layers allow different types of content (terrain, objects, water, effects)
//! to be streamed, updated, and rendered independently.

pub mod compositor;

pub use compositor::LayerCompositor;

/// Layer rendering mode affects compositing order and shader selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerRenderMode {
    /// Standard opaque rendering (terrain, rocks, buildings)
    Opaque,
    /// Alpha-tested transparency (leaves, grass - binary transparency)
    AlphaTest { threshold: u8 },
    /// Full transparency with sorted rendering (glass, water surface)
    Transparent,
    /// Volumetric rendering (underwater fog, clouds, particles)
    Volumetric,
}

/// Update frequency hint for streaming priority and caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpdateFrequency {
    /// Rarely changes (terrain base) - heavily cached, lowest streaming priority
    Static,
    /// Occasionally changes (buildings, trees) - edit support, medium priority
    Dynamic,
    /// Per-frame updates (animated characters, particles) - highest priority
    PerFrame,
}

/// Unique identifier for a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerId(pub u32);

impl LayerId {
    /// Base terrain layer - static, opaque
    pub const TERRAIN: LayerId = LayerId(0);
    /// Static objects (trees, buildings) - dynamic, opaque
    pub const STATIC_OBJECTS: LayerId = LayerId(1);
    /// Dynamic objects (characters, vehicles) - per-frame, opaque
    pub const DYNAMIC_OBJECTS: LayerId = LayerId(2);
    /// Water layer - dynamic, transparent
    pub const WATER: LayerId = LayerId(3);
    /// Effects layer (particles, fog) - per-frame, volumetric
    pub const EFFECTS: LayerId = LayerId(4);
    /// Ground clutter layer (rocks, sticks, fallen logs) - static, opaque
    pub const GROUND_CLUTTER: LayerId = LayerId(5);
}

/// Configuration for a render layer.
#[derive(Clone, Debug)]
pub struct LayerConfig {
    /// Unique identifier
    pub id: LayerId,
    /// Human-readable name
    pub name: String,
    /// Rendering mode (opaque, transparent, volumetric)
    pub render_mode: LayerRenderMode,
    /// Update frequency hint
    pub update_frequency: UpdateFrequency,
    /// Render priority (lower = rendered first)
    pub priority: i32,
    /// Fraction of streaming budget allocated to this layer (0.0-1.0)
    pub streaming_budget_fraction: f32,
    /// Whether this layer is currently enabled
    pub enabled: bool,
}

impl LayerConfig {
    /// Create a new layer configuration.
    pub fn new(id: LayerId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            render_mode: LayerRenderMode::Opaque,
            update_frequency: UpdateFrequency::Static,
            priority: id.0 as i32,
            streaming_budget_fraction: 0.2,
            enabled: true,
        }
    }

    /// Set the render mode.
    pub fn with_render_mode(mut self, mode: LayerRenderMode) -> Self {
        self.render_mode = mode;
        self
    }

    /// Set the update frequency.
    pub fn with_update_frequency(mut self, freq: UpdateFrequency) -> Self {
        self.update_frequency = freq;
        self
    }

    /// Set the render priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the streaming budget fraction.
    pub fn with_streaming_budget(mut self, fraction: f32) -> Self {
        self.streaming_budget_fraction = fraction.clamp(0.0, 1.0);
        self
    }
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self::new(LayerId(0), "default")
    }
}

/// Default layer configurations for common use cases.
pub fn default_layers() -> Vec<LayerConfig> {
    vec![
        LayerConfig::new(LayerId::TERRAIN, "Terrain")
            .with_render_mode(LayerRenderMode::Opaque)
            .with_update_frequency(UpdateFrequency::Static)
            .with_priority(0)
            .with_streaming_budget(0.35),
        LayerConfig::new(LayerId::STATIC_OBJECTS, "Static Objects")
            .with_render_mode(LayerRenderMode::Opaque)
            .with_update_frequency(UpdateFrequency::Dynamic)
            .with_priority(1)
            .with_streaming_budget(0.20),
        LayerConfig::new(LayerId::DYNAMIC_OBJECTS, "Dynamic Objects")
            .with_render_mode(LayerRenderMode::Opaque)
            .with_update_frequency(UpdateFrequency::PerFrame)
            .with_priority(2)
            .with_streaming_budget(0.10),
        LayerConfig::new(LayerId::GROUND_CLUTTER, "Ground Clutter")
            .with_render_mode(LayerRenderMode::Opaque)
            .with_update_frequency(UpdateFrequency::Static)
            .with_priority(1) // Render with static objects
            .with_streaming_budget(0.15),
        LayerConfig::new(LayerId::WATER, "Water")
            .with_render_mode(LayerRenderMode::Transparent)
            .with_update_frequency(UpdateFrequency::Dynamic)
            .with_priority(10) // Render after opaque
            .with_streaming_budget(0.10),
        LayerConfig::new(LayerId::EFFECTS, "Effects")
            .with_render_mode(LayerRenderMode::Volumetric)
            .with_update_frequency(UpdateFrequency::PerFrame)
            .with_priority(20) // Render last
            .with_streaming_budget(0.10),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_ids_are_unique() {
        let ids = [
            LayerId::TERRAIN,
            LayerId::STATIC_OBJECTS,
            LayerId::DYNAMIC_OBJECTS,
            LayerId::WATER,
            LayerId::EFFECTS,
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "Layer IDs must be unique");
            }
        }
    }

    #[test]
    fn test_default_layers() {
        let layers = default_layers();
        assert_eq!(layers.len(), 6);

        // Budget should sum to ~1.0
        let total_budget: f32 = layers.iter().map(|l| l.streaming_budget_fraction).sum();
        assert!((total_budget - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_config_builder() {
        let config = LayerConfig::new(LayerId(99), "Custom")
            .with_render_mode(LayerRenderMode::Transparent)
            .with_update_frequency(UpdateFrequency::PerFrame)
            .with_priority(50)
            .with_streaming_budget(0.5);

        assert_eq!(config.id, LayerId(99));
        assert_eq!(config.name, "Custom");
        assert_eq!(config.render_mode, LayerRenderMode::Transparent);
        assert_eq!(config.update_frequency, UpdateFrequency::PerFrame);
        assert_eq!(config.priority, 50);
        assert_eq!(config.streaming_budget_fraction, 0.5);
    }
}
