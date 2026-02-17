//! Layer compositor for multi-layer voxel rendering.
//!
//! Coordinates rendering order, streaming budgets, and provides
//! combined classification for octree building.

use std::collections::HashMap;

use super::{LayerConfig, LayerId, LayerRenderMode, UpdateFrequency};

/// Manages multiple voxel layers for coordinated rendering and streaming.
pub struct LayerCompositor {
    /// Layer configurations by ID
    layers: HashMap<LayerId, LayerConfig>,
    /// Render order (computed from layer priorities)
    render_order: Vec<LayerId>,
}

impl LayerCompositor {
    /// Create a new empty compositor.
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            render_order: Vec::new(),
        }
    }

    /// Create with default layers.
    pub fn with_default_layers() -> Self {
        let mut compositor = Self::new();
        for config in super::default_layers() {
            compositor.add_layer(config);
        }
        compositor
    }

    /// Add a layer configuration.
    pub fn add_layer(&mut self, config: LayerConfig) {
        self.layers.insert(config.id, config);
        self.rebuild_render_order();
    }

    /// Remove a layer.
    pub fn remove_layer(&mut self, id: LayerId) -> Option<LayerConfig> {
        let config = self.layers.remove(&id);
        if config.is_some() {
            self.rebuild_render_order();
        }
        config
    }

    /// Get a layer configuration.
    pub fn get_layer(&self, id: LayerId) -> Option<&LayerConfig> {
        self.layers.get(&id)
    }

    /// Get mutable layer configuration.
    pub fn get_layer_mut(&mut self, id: LayerId) -> Option<&mut LayerConfig> {
        self.layers.get_mut(&id)
    }

    /// Enable or disable a layer.
    pub fn set_layer_enabled(&mut self, id: LayerId, enabled: bool) {
        if let Some(config) = self.layers.get_mut(&id) {
            config.enabled = enabled;
        }
    }

    /// Iterate layers in render order (opaque first, then transparent, then volumetric).
    pub fn render_order(&self) -> impl Iterator<Item = &LayerConfig> {
        self.render_order
            .iter()
            .filter_map(|id| self.layers.get(id))
            .filter(|config| config.enabled)
    }

    /// Get opaque layers only.
    pub fn opaque_layers(&self) -> impl Iterator<Item = &LayerConfig> {
        self.render_order().filter(|c| c.render_mode == LayerRenderMode::Opaque)
    }

    /// Get transparent layers only.
    pub fn transparent_layers(&self) -> impl Iterator<Item = &LayerConfig> {
        self.render_order().filter(|c| {
            matches!(c.render_mode, LayerRenderMode::Transparent | LayerRenderMode::AlphaTest { .. })
        })
    }

    /// Get volumetric layers only.
    pub fn volumetric_layers(&self) -> impl Iterator<Item = &LayerConfig> {
        self.render_order().filter(|c| c.render_mode == LayerRenderMode::Volumetric)
    }

    /// Get layers that need per-frame updates.
    pub fn per_frame_layers(&self) -> impl Iterator<Item = &LayerConfig> {
        self.layers.values().filter(|c| c.enabled && c.update_frequency == UpdateFrequency::PerFrame)
    }

    /// Calculate streaming budget for a layer based on total budget.
    pub fn streaming_budget_for_layer(&self, id: LayerId, total_budget_bytes: usize) -> usize {
        self.layers
            .get(&id)
            .map(|c| (total_budget_bytes as f32 * c.streaming_budget_fraction) as usize)
            .unwrap_or(0)
    }

    /// Total number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Rebuild render order from layer priorities.
    fn rebuild_render_order(&mut self) {
        let mut order: Vec<_> = self.layers.values().collect();

        // Sort by: render mode category first, then priority within category
        order.sort_by(|a, b| {
            let mode_order = |m: &LayerRenderMode| match m {
                LayerRenderMode::Opaque => 0,
                LayerRenderMode::AlphaTest { .. } => 1,
                LayerRenderMode::Transparent => 2,
                LayerRenderMode::Volumetric => 3,
            };

            let a_mode = mode_order(&a.render_mode);
            let b_mode = mode_order(&b.render_mode);

            if a_mode != b_mode {
                a_mode.cmp(&b_mode)
            } else {
                a.priority.cmp(&b.priority)
            }
        });

        self.render_order = order.into_iter().map(|c| c.id).collect();
    }
}

impl Default for LayerCompositor {
    fn default() -> Self {
        Self::with_default_layers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_compositor() {
        let compositor = LayerCompositor::default();
        assert_eq!(compositor.layer_count(), 6);
    }

    #[test]
    fn test_render_order() {
        let compositor = LayerCompositor::default();
        let order: Vec<_> = compositor.render_order().collect();

        // Opaque layers should come first
        assert!(order.iter().take_while(|c| c.render_mode == LayerRenderMode::Opaque).count() >= 1);
    }

    #[test]
    fn test_streaming_budget() {
        let compositor = LayerCompositor::default();
        let total = 1000;

        let terrain_budget = compositor.streaming_budget_for_layer(LayerId::TERRAIN, total);
        assert_eq!(terrain_budget, 350); // 35% of 1000
    }

    #[test]
    fn test_disable_layer() {
        let mut compositor = LayerCompositor::default();
        compositor.set_layer_enabled(LayerId::WATER, false);

        let enabled: Vec<_> = compositor.render_order().collect();
        assert!(!enabled.iter().any(|c| c.id == LayerId::WATER));
    }
}
