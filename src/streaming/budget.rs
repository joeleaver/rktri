//! Memory budget management for streaming
//!
//! Tracks CPU and GPU memory usage to prevent out-of-memory conditions.
//! Provides pressure metrics to guide eviction and loading decisions.

/// Memory budget manager
///
/// Tracks CPU and GPU memory usage and provides pressure metrics to guide
/// chunk loading and eviction decisions.
pub struct MemoryBudget {
    /// Maximum CPU memory allowed (bytes)
    cpu_budget_bytes: usize,
    /// Maximum GPU memory allowed (bytes)
    gpu_budget_bytes: usize,
    /// Currently used CPU memory (bytes)
    cpu_used_bytes: usize,
    /// Currently used GPU memory (bytes)
    gpu_used_bytes: usize,
}

impl MemoryBudget {
    /// Create a new memory budget
    ///
    /// # Arguments
    /// * `cpu_budget_mb` - Maximum CPU memory in megabytes
    /// * `gpu_budget_mb` - Maximum GPU memory in megabytes
    pub fn new(cpu_budget_mb: usize, gpu_budget_mb: usize) -> Self {
        Self {
            cpu_budget_bytes: cpu_budget_mb * 1024 * 1024,
            gpu_budget_bytes: gpu_budget_mb * 1024 * 1024,
            cpu_used_bytes: 0,
            gpu_used_bytes: 0,
        }
    }

    // --- Tracking methods ---

    /// Add CPU memory usage
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to add
    pub fn add_cpu(&mut self, bytes: usize) {
        self.cpu_used_bytes = self.cpu_used_bytes.saturating_add(bytes);
    }

    /// Add GPU memory usage
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to add
    pub fn add_gpu(&mut self, bytes: usize) {
        self.gpu_used_bytes = self.gpu_used_bytes.saturating_add(bytes);
    }

    /// Remove CPU memory usage
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to remove
    pub fn remove_cpu(&mut self, bytes: usize) {
        self.cpu_used_bytes = self.cpu_used_bytes.saturating_sub(bytes);
    }

    /// Remove GPU memory usage
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to remove
    pub fn remove_gpu(&mut self, bytes: usize) {
        self.gpu_used_bytes = self.gpu_used_bytes.saturating_sub(bytes);
    }

    // --- Query methods ---

    /// Get current CPU memory usage in bytes
    pub fn cpu_used(&self) -> usize {
        self.cpu_used_bytes
    }

    /// Get current GPU memory usage in bytes
    pub fn gpu_used(&self) -> usize {
        self.gpu_used_bytes
    }

    /// Get available CPU memory in bytes
    pub fn cpu_available(&self) -> usize {
        self.cpu_budget_bytes.saturating_sub(self.cpu_used_bytes)
    }

    /// Get available GPU memory in bytes
    pub fn gpu_available(&self) -> usize {
        self.gpu_budget_bytes.saturating_sub(self.gpu_used_bytes)
    }

    /// Get CPU memory pressure (0.0 to 1.0+)
    ///
    /// Values above 0.9 indicate high pressure.
    /// Values above 1.0 indicate over-budget.
    pub fn cpu_pressure(&self) -> f32 {
        if self.cpu_budget_bytes == 0 {
            return 0.0;
        }
        self.cpu_used_bytes as f32 / self.cpu_budget_bytes as f32
    }

    /// Get GPU memory pressure (0.0 to 1.0+)
    ///
    /// Values above 0.9 indicate high pressure.
    /// Values above 1.0 indicate over-budget.
    pub fn gpu_pressure(&self) -> f32 {
        if self.gpu_budget_bytes == 0 {
            return 0.0;
        }
        self.gpu_used_bytes as f32 / self.gpu_budget_bytes as f32
    }

    // --- Decision methods ---

    /// Check if we should evict chunks
    ///
    /// Returns true if either CPU or GPU pressure exceeds 0.9 (90% usage)
    pub fn should_evict(&self) -> bool {
        self.cpu_pressure() > 0.9 || self.gpu_pressure() > 0.9
    }

    /// Check if we can load a chunk with the given memory requirements
    ///
    /// # Arguments
    /// * `cpu_bytes` - CPU memory required
    /// * `gpu_bytes` - GPU memory required
    ///
    /// # Returns
    /// True if there's enough available memory for both CPU and GPU
    pub fn can_load(&self, cpu_bytes: usize, gpu_bytes: usize) -> bool {
        self.cpu_available() >= cpu_bytes && self.gpu_available() >= gpu_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_new() {
        let budget = MemoryBudget::new(512, 1024);
        assert_eq!(budget.cpu_used(), 0);
        assert_eq!(budget.gpu_used(), 0);
        assert_eq!(budget.cpu_available(), 512 * 1024 * 1024);
        assert_eq!(budget.gpu_available(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_budget_add_cpu() {
        let mut budget = MemoryBudget::new(512, 1024);
        budget.add_cpu(100 * 1024 * 1024); // 100 MB

        assert_eq!(budget.cpu_used(), 100 * 1024 * 1024);
        assert_eq!(budget.cpu_available(), 412 * 1024 * 1024);
    }

    #[test]
    fn test_budget_add_gpu() {
        let mut budget = MemoryBudget::new(512, 1024);
        budget.add_gpu(200 * 1024 * 1024); // 200 MB

        assert_eq!(budget.gpu_used(), 200 * 1024 * 1024);
        assert_eq!(budget.gpu_available(), 824 * 1024 * 1024);
    }

    #[test]
    fn test_budget_remove_cpu() {
        let mut budget = MemoryBudget::new(512, 1024);
        budget.add_cpu(100 * 1024 * 1024);
        budget.remove_cpu(50 * 1024 * 1024);

        assert_eq!(budget.cpu_used(), 50 * 1024 * 1024);
        assert_eq!(budget.cpu_available(), 462 * 1024 * 1024);
    }

    #[test]
    fn test_budget_remove_gpu() {
        let mut budget = MemoryBudget::new(512, 1024);
        budget.add_gpu(200 * 1024 * 1024);
        budget.remove_gpu(100 * 1024 * 1024);

        assert_eq!(budget.gpu_used(), 100 * 1024 * 1024);
        assert_eq!(budget.gpu_available(), 924 * 1024 * 1024);
    }

    #[test]
    fn test_budget_saturating_ops() {
        let mut budget = MemoryBudget::new(512, 1024);

        // Removing more than available should saturate at 0
        budget.remove_cpu(1000 * 1024 * 1024);
        assert_eq!(budget.cpu_used(), 0);

        // Adding beyond usize::MAX should saturate
        budget.add_cpu(usize::MAX);
        budget.add_cpu(100);
        assert_eq!(budget.cpu_used(), usize::MAX); // Should saturate
    }

    #[test]
    fn test_budget_cpu_pressure() {
        let mut budget = MemoryBudget::new(100, 100); // 100 MB each

        // 0% pressure
        assert_eq!(budget.cpu_pressure(), 0.0);

        // 50% pressure
        budget.add_cpu(50 * 1024 * 1024);
        assert!((budget.cpu_pressure() - 0.5).abs() < 0.01);

        // 90% pressure
        budget.add_cpu(40 * 1024 * 1024);
        assert!((budget.cpu_pressure() - 0.9).abs() < 0.01);

        // 100% pressure
        budget.add_cpu(10 * 1024 * 1024);
        assert!((budget.cpu_pressure() - 1.0).abs() < 0.01);

        // Over-budget (110%)
        budget.add_cpu(10 * 1024 * 1024);
        assert!(budget.cpu_pressure() > 1.0);
    }

    #[test]
    fn test_budget_gpu_pressure() {
        let mut budget = MemoryBudget::new(100, 200); // 200 MB GPU

        // 0% pressure
        assert_eq!(budget.gpu_pressure(), 0.0);

        // 50% pressure
        budget.add_gpu(100 * 1024 * 1024);
        assert!((budget.gpu_pressure() - 0.5).abs() < 0.01);

        // 95% pressure
        budget.add_gpu(90 * 1024 * 1024);
        assert!((budget.gpu_pressure() - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_budget_zero_budget_pressure() {
        let budget = MemoryBudget::new(0, 0);
        assert_eq!(budget.cpu_pressure(), 0.0);
        assert_eq!(budget.gpu_pressure(), 0.0);
    }

    #[test]
    fn test_budget_should_evict() {
        let mut budget = MemoryBudget::new(100, 100);

        // Low pressure - should not evict
        budget.add_cpu(50 * 1024 * 1024);
        assert!(!budget.should_evict());

        // High CPU pressure - should evict
        budget.add_cpu(45 * 1024 * 1024); // 95% CPU usage
        assert!(budget.should_evict());
    }

    #[test]
    fn test_budget_should_evict_gpu() {
        let mut budget = MemoryBudget::new(100, 100);

        // Low pressure - should not evict
        budget.add_gpu(50 * 1024 * 1024);
        assert!(!budget.should_evict());

        // High GPU pressure - should evict
        budget.add_gpu(45 * 1024 * 1024); // 95% GPU usage
        assert!(budget.should_evict());
    }

    #[test]
    fn test_budget_can_load() {
        let mut budget = MemoryBudget::new(100, 100);

        // Can load when plenty of space
        assert!(budget.can_load(10 * 1024 * 1024, 10 * 1024 * 1024));

        // Use up some memory
        budget.add_cpu(95 * 1024 * 1024);
        budget.add_gpu(95 * 1024 * 1024);

        // Cannot load when insufficient space
        assert!(!budget.can_load(10 * 1024 * 1024, 10 * 1024 * 1024));

        // Can load smaller amount
        assert!(budget.can_load(4 * 1024 * 1024, 4 * 1024 * 1024));
    }

    #[test]
    fn test_budget_can_load_cpu_only() {
        let mut budget = MemoryBudget::new(100, 100);
        budget.add_cpu(95 * 1024 * 1024);

        // Cannot load due to CPU constraint
        assert!(!budget.can_load(10 * 1024 * 1024, 1 * 1024 * 1024));
    }

    #[test]
    fn test_budget_can_load_gpu_only() {
        let mut budget = MemoryBudget::new(100, 100);
        budget.add_gpu(95 * 1024 * 1024);

        // Cannot load due to GPU constraint
        assert!(!budget.can_load(1 * 1024 * 1024, 10 * 1024 * 1024));
    }

    #[test]
    fn test_budget_realistic_scenario() {
        // Realistic scenario: 2GB CPU, 4GB GPU
        let mut budget = MemoryBudget::new(2048, 4096);

        // Load some chunks (assume 50MB CPU, 100MB GPU each)
        for _ in 0..10 {
            budget.add_cpu(50 * 1024 * 1024);
            budget.add_gpu(100 * 1024 * 1024);
        }

        // Check usage
        assert_eq!(budget.cpu_used(), 500 * 1024 * 1024); // 500 MB
        assert_eq!(budget.gpu_used(), 1000 * 1024 * 1024); // 1000 MB

        // Check pressure
        assert!((budget.cpu_pressure() - 0.244).abs() < 0.01); // ~24.4%
        assert!((budget.gpu_pressure() - 0.244).abs() < 0.01); // ~24.4%

        // Should not evict yet
        assert!(!budget.should_evict());

        // Can still load more
        assert!(budget.can_load(50 * 1024 * 1024, 100 * 1024 * 1024));
    }
}
