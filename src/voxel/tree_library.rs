//! Tree library management for pre-generated tree assets

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::voxel::procgen::TreeStyle;
use crate::voxel::tree_data::{TreeData, TREE_FILE_EXTENSION};

/// Entry in the tree library
#[derive(Debug, Clone)]
pub struct TreeEntry {
    /// Unique ID within library
    pub id: u32,
    /// Tree style
    pub style: TreeStyle,
    /// Generation seed
    pub seed: u64,
    /// File path relative to library root
    pub path: PathBuf,
    /// Bounding box min corner
    pub bounds_min: [f32; 3],
    /// Bounding box max corner
    pub bounds_max: [f32; 3],
}

/// Index file data (serialized as JSON for easy inspection)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LibraryIndex {
    version: u32,
    entries: Vec<IndexEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexEntry {
    id: u32,
    style: String,
    seed: u64,
    path: String,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

const INDEX_VERSION: u32 = 1;
const INDEX_FILENAME: &str = "index.json";

fn style_from_str(s: &str) -> TreeStyle {
    match s {
        "Oak" => TreeStyle::Oak,
        "Willow" => TreeStyle::Willow,
        "Elm" => TreeStyle::Elm,
        _ => TreeStyle::Oak,
    }
}

fn style_to_str(style: TreeStyle) -> &'static str {
    match style {
        TreeStyle::Oak => "Oak",
        TreeStyle::Willow => "Willow",
        TreeStyle::Elm => "Elm",
    }
}

fn style_to_dir(style: TreeStyle) -> &'static str {
    match style {
        TreeStyle::Oak => "oak",
        TreeStyle::Willow => "willow",
        TreeStyle::Elm => "elm",
    }
}

/// Tree library manager
pub struct TreeLibrary {
    /// Base directory for tree files
    base_dir: PathBuf,
    /// All tree entries indexed by ID
    entries: HashMap<u32, TreeEntry>,
    /// Style -> IDs index for filtered queries
    by_style: HashMap<TreeStyle, Vec<u32>>,
    /// Next ID to assign
    next_id: u32,
}

impl TreeLibrary {
    /// Create a new empty library at the given path
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            entries: HashMap::new(),
            by_style: HashMap::new(),
            next_id: 1,
        }
    }

    fn load_from_index(base_dir: PathBuf, index: LibraryIndex) -> Self {
        let mut library = Self::new(base_dir);

        for entry in index.entries {
            let style = style_from_str(&entry.style);

            let tree_entry = TreeEntry {
                id: entry.id,
                style,
                seed: entry.seed,
                path: PathBuf::from(entry.path),
                bounds_min: entry.bounds_min,
                bounds_max: entry.bounds_max,
            };

            library.by_style.entry(style).or_default().push(entry.id);
            library.entries.insert(entry.id, tree_entry);
            library.next_id = library.next_id.max(entry.id + 1);
        }

        library
    }

    /// Open an existing library or create a new one
    pub async fn open(base_dir: PathBuf) -> Result<Self, io::Error> {
        let index_path = base_dir.join(INDEX_FILENAME);

        if index_path.exists() {
            let index_data = tokio::fs::read_to_string(&index_path).await?;
            let index: LibraryIndex = serde_json::from_str(&index_data)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Self::load_from_index(base_dir, index))
        } else {
            tokio::fs::create_dir_all(&base_dir).await?;
            Ok(Self::new(base_dir))
        }
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

    fn add_entry(&mut self, data: &TreeData) -> (u32, PathBuf) {
        let id = self.next_id;
        self.next_id += 1;

        let style = data.style();
        let dir_name = style_to_dir(style);

        let filename = format!("{}_{:04}.{}", dir_name, id, TREE_FILE_EXTENSION);
        let rel_path = PathBuf::from(dir_name).join(&filename);

        let entry = TreeEntry {
            id,
            style,
            seed: data.seed,
            path: rel_path.clone(),
            bounds_min: data.bounds_min,
            bounds_max: data.bounds_max,
        };

        self.by_style.entry(style).or_default().push(id);
        self.entries.insert(id, entry);

        (id, rel_path)
    }

    /// Add a tree to the library
    pub async fn add_tree(&mut self, data: &TreeData) -> Result<u32, io::Error> {
        let (id, rel_path) = self.add_entry(data);
        let full_path = self.base_dir.join(&rel_path);

        data.save(&full_path).await?;
        self.save_index().await?;

        Ok(id)
    }

    /// Add a tree to the library (sync)
    pub fn add_tree_sync(&mut self, data: &TreeData) -> Result<u32, io::Error> {
        let (id, rel_path) = self.add_entry(data);
        let full_path = self.base_dir.join(&rel_path);

        data.save_sync(&full_path)?;
        self.save_index_sync()?;

        Ok(id)
    }

    /// Load tree data by ID
    pub async fn load(&self, id: u32) -> Result<TreeData, io::Error> {
        let entry = self.entries.get(&id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Tree {} not found", id)))?;

        let full_path = self.base_dir.join(&entry.path);
        TreeData::load(&full_path).await
    }

    /// Load tree data by ID (sync)
    pub fn load_sync(&self, id: u32) -> Result<TreeData, io::Error> {
        let entry = self.entries.get(&id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Tree {} not found", id)))?;

        let full_path = self.base_dir.join(&entry.path);
        TreeData::load_sync(&full_path)
    }

    /// Get a random tree ID by style
    pub fn random_by_style(&self, style: TreeStyle, seed: u64) -> Option<u32> {
        let ids = self.by_style.get(&style)?;
        if ids.is_empty() {
            return None;
        }
        // Simple deterministic selection based on seed
        let index = (seed as usize) % ids.len();
        Some(ids[index])
    }

    /// Get all tree IDs for a style
    pub fn ids_by_style(&self, style: TreeStyle) -> &[u32] {
        self.by_style.get(&style).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get tree entry by ID
    pub fn entry(&self, id: u32) -> Option<&TreeEntry> {
        self.entries.get(&id)
    }

    /// Get total number of trees
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if library is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get count by style
    pub fn count_by_style(&self, style: TreeStyle) -> usize {
        self.by_style.get(&style).map(|v| v.len()).unwrap_or(0)
    }

    /// Get base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Save library index
    pub async fn save_index(&self) -> Result<(), io::Error> {
        let index = self.build_index();
        let json = serde_json::to_string_pretty(&index)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        tokio::fs::write(self.base_dir.join(INDEX_FILENAME), json).await
    }

    /// Save library index (sync)
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
                style: style_to_str(e.style).to_string(),
                seed: e.seed,
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
    use crate::voxel::procgen::TreeGenerator;
    use tempfile::TempDir;

    #[test]
    fn test_library_create_and_add() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let mut library = TreeLibrary::new(temp_dir.path().to_path_buf());

        // Generate a tree
        let mut generator = TreeGenerator::from_style(42, TreeStyle::Oak);
        let octree = generator.generate(8.0, 5);
        let tree_data = TreeData::from_octree(&octree, TreeStyle::Oak, 42);

        // Add to library
        let id = library.add_tree_sync(&tree_data).expect("add failed");
        assert_eq!(id, 1);
        assert_eq!(library.len(), 1);
        assert_eq!(library.count_by_style(TreeStyle::Oak), 1);
    }

    #[test]
    fn test_library_load() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let mut library = TreeLibrary::new(temp_dir.path().to_path_buf());

        let mut generator = TreeGenerator::from_style(123, TreeStyle::Elm);
        let octree = generator.generate(8.0, 5);
        let tree_data = TreeData::from_octree(&octree, TreeStyle::Elm, 123);

        let id = library.add_tree_sync(&tree_data).expect("add failed");

        // Load back
        let loaded = library.load_sync(id).expect("load failed");
        assert_eq!(loaded.seed, 123);
        assert_eq!(loaded.style(), TreeStyle::Elm);
    }

    #[test]
    fn test_library_persistence() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let path = temp_dir.path().to_path_buf();

        // Create and add tree
        {
            let mut library = TreeLibrary::new(path.clone());
            let mut generator = TreeGenerator::from_style(999, TreeStyle::Willow);
            let octree = generator.generate(8.0, 5);
            let tree_data = TreeData::from_octree(&octree, TreeStyle::Willow, 999);
            library.add_tree_sync(&tree_data).expect("add failed");
        }

        // Reopen and verify
        {
            let library = TreeLibrary::open_sync(path).expect("open failed");
            assert_eq!(library.len(), 1);
            assert_eq!(library.count_by_style(TreeStyle::Willow), 1);
        }
    }

    #[test]
    fn test_random_selection() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let mut library = TreeLibrary::new(temp_dir.path().to_path_buf());

        // Add multiple trees
        for i in 0..5 {
            let mut generator = TreeGenerator::from_style(i, TreeStyle::Oak);
            let octree = generator.generate(8.0, 5);
            let tree_data = TreeData::from_octree(&octree, TreeStyle::Oak, i);
            library.add_tree_sync(&tree_data).expect("add failed");
        }

        // Random selection should work
        let id1 = library.random_by_style(TreeStyle::Oak, 42);
        let id2 = library.random_by_style(TreeStyle::Oak, 43);

        assert!(id1.is_some());
        assert!(id2.is_some());
    }
}
