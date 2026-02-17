//! Clutter library management for pre-generated clutter voxel objects.
//!
//! This module manages the clutter library index. The actual voxel data
//! for clutter objects is stored separately and referenced by ID.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Current version of clutter data format
pub const CLUTTER_DATA_VERSION: u32 = 1;

/// File extension for clutter data files
pub const CLUTTER_FILE_EXTENSION: &str = "rkcld";

/// Clutter object metadata (lightweight, stored in library index)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClutterData {
    /// Format version for compatibility
    pub version: u32,
    /// Unique object type ID within library
    pub object_id: u16,
    /// Object name (e.g., "small_rock", "stick")
    pub name: String,
    /// Bounding box min corner (relative to object origin)
    pub bounds_min: [f32; 3],
    /// Bounding box max corner
    pub bounds_max: [f32; 3],
    /// Root size in world units
    pub root_size: f32,
    /// Maximum octree depth
    pub max_depth: u8,
    /// Node count
    pub node_count: u32,
    /// Brick count
    pub brick_count: u32,
}

impl ClutterData {
    /// Create new clutter data metadata
    pub fn new(object_id: u16, name: impl Into<String>, bounds_min: [f32; 3], bounds_max: [f32; 3], root_size: f32, max_depth: u8, node_count: u32, brick_count: u32) -> Self {
        Self {
            version: CLUTTER_DATA_VERSION,
            object_id,
            name: name.into(),
            bounds_min,
            bounds_max,
            root_size,
            max_depth,
            node_count,
            brick_count,
        }
    }

    /// Get bounding box dimensions
    pub fn bounds_size(&self) -> [f32; 3] {
        [
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        ]
    }

    /// Save to file (sync)
    pub fn save_sync(&self, path: &Path) -> Result<(), io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, json)
    }

    /// Load from file (sync)
    pub fn load_sync(path: &Path) -> Result<Self, io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
}

/// Entry in the clutter library
#[derive(Debug, Clone)]
pub struct ClutterEntry {
    /// Unique ID within library
    pub id: u16,
    /// Object name
    pub name: String,
    /// File path relative to library root
    pub path: PathBuf,
    /// Bounding box min corner
    pub bounds_min: [f32; 3],
    /// Bounding box max corner
    pub bounds_max: [f32; 3],
}

/// Index file data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LibraryIndex {
    version: u32,
    entries: Vec<IndexEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexEntry {
    id: u16,
    name: String,
    path: String,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

const INDEX_VERSION: u32 = 1;
const INDEX_FILENAME: &str = "index.json";

/// Clutter library manager
pub struct ClutterLibrary {
    /// Base directory for clutter files
    base_dir: PathBuf,
    /// All clutter entries indexed by ID
    entries: HashMap<u16, ClutterEntry>,
    /// Name -> ID lookup
    by_name: HashMap<String, u16>,
    /// Next ID to assign
    next_id: u16,
}

impl ClutterLibrary {
    /// Create a new empty library at the given path
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            entries: HashMap::new(),
            by_name: HashMap::new(),
            next_id: 1,
        }
    }

    fn load_from_index(base_dir: PathBuf, index: LibraryIndex) -> Self {
        let mut library = Self::new(base_dir);

        for entry in index.entries {
            let clutter_entry = ClutterEntry {
                id: entry.id,
                name: entry.name.clone(),
                path: PathBuf::from(entry.path),
                bounds_min: entry.bounds_min,
                bounds_max: entry.bounds_max,
            };

            library.by_name.insert(entry.name, entry.id);
            library.entries.insert(entry.id, clutter_entry);
            library.next_id = library.next_id.max(entry.id + 1);
        }

        library
    }

    /// Open an existing library or create a new one (sync)
    pub fn open_sync(base_dir: PathBuf) -> Result<Self, io::Error> {
        let index_path = base_dir.join(INDEX_FILENAME);

        if index_path.exists() {
            let index_data = std::fs::read_to_string(&index_path)?;
            let index: LibraryIndex = serde_json::from_str(&index_data)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Self::load_from_index(base_dir, index))
        } else {
            std::fs::create_dir_all(&base_dir)?;
            Ok(Self::new(base_dir))
        }
    }

    fn add_entry(&mut self, data: &ClutterData) -> (u16, PathBuf) {
        let id = self.next_id;
        self.next_id += 1;

        let filename = format!("{}_{:04}.{}", data.name, id, CLUTTER_FILE_EXTENSION);
        let rel_path = PathBuf::from(&data.name).join(&filename);

        let entry = ClutterEntry {
            id,
            name: data.name.clone(),
            path: rel_path.clone(),
            bounds_min: data.bounds_min,
            bounds_max: data.bounds_max,
        };

        self.by_name.insert(data.name.clone(), id);
        self.entries.insert(id, entry);

        (id, rel_path)
    }

    /// Add clutter data to the library
    pub fn add_clutter(&mut self, data: &ClutterData) -> Result<u16, io::Error> {
        let (id, rel_path) = self.add_entry(data);
        let full_path = self.base_dir.join(&rel_path);

        data.save_sync(&full_path)?;
        self.save_index_sync()?;

        Ok(id)
    }

    /// Load clutter data by ID
    pub fn load(&self, id: u16) -> Result<ClutterData, io::Error> {
        let entry = self.entries.get(&id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Clutter {} not found", id)))?;

        let full_path = self.base_dir.join(&entry.path);
        ClutterData::load_sync(&full_path)
    }

    /// Get clutter entry by ID
    pub fn entry(&self, id: u16) -> Option<&ClutterEntry> {
        self.entries.get(&id)
    }

    /// Get clutter by name
    pub fn by_name(&self, name: &str) -> Option<u16> {
        self.by_name.get(name).copied()
    }

    /// Get total number of objects
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if library is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Save library index
    pub fn save_index_sync(&self) -> Result<(), io::Error> {
        let index = self.build_index();
        let json = serde_json::to_string_pretty(&index)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(self.base_dir.join(INDEX_FILENAME), json)
    }

    fn build_index(&self) -> LibraryIndex {
        let entries: Vec<IndexEntry> = self.entries.values().map(|e| {
            IndexEntry {
                id: e.id,
                name: e.name.clone(),
                path: e.path.to_string_lossy().to_string(),
                bounds_min: e.bounds_min,
                bounds_max: e.bounds_max,
            }
        }).collect();

        LibraryIndex {
            version: INDEX_VERSION,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_library_create() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let library = ClutterLibrary::new(temp_dir.path().to_path_buf());
        assert!(library.is_empty());
    }

    #[test]
    fn test_library_persistence() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let path = temp_dir.path().to_path_buf();

        // Create and add clutter
        {
            let mut library = ClutterLibrary::new(path.clone());
            let data = ClutterData::new(1, "test_rock", [-0.5, 0.0, -0.5], [0.5, 0.5, 0.5], 1.0, 5, 100, 10);
            library.add_clutter(&data).expect("add failed");
        }

        // Reopen and verify
        {
            let library = ClutterLibrary::open_sync(path).expect("open failed");
            assert_eq!(library.len(), 1);
        }
    }

    #[test]
    fn test_clutter_data_bounds() {
        let data = ClutterData::new(
            1, "test", 
            [-0.5, 0.0, -0.5], 
            [0.5, 1.0, 0.5], 
            1.0, 5, 100, 10
        );
        assert_eq!(data.bounds_size(), [1.0, 1.0, 1.0]);
    }
}
