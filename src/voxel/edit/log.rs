//! Append-only log of edits for persistence.

use std::collections::HashMap;
use std::io::{self, BufWriter, Write, BufReader, Read};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::types::Vec3;
use crate::math::Aabb;
use crate::voxel::chunk::ChunkCoord;
use crate::voxel::voxel::Voxel;
use super::delta::{EditDelta, EditOp};

const MAGIC: &[u8; 4] = b"RKED";
const VERSION: u32 = 1;

/// Append-only log of edits for persistence.
pub struct EditLog {
    /// Path to the log file
    path: PathBuf,
    /// In-memory index: chunk -> edit IDs
    chunk_index: HashMap<ChunkCoord, Vec<u64>>,
    /// All edits in memory
    edits: Vec<EditDelta>,
    /// Next edit ID
    next_id: AtomicU64,
}

impl EditLog {
    /// Create new or open existing log.
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();

        // Try to load existing, otherwise create empty
        if path.exists() {
            Self::load(&path).unwrap_or_else(|_| Self::empty(path))
        } else {
            Self::empty(path)
        }
    }

    /// Create empty log.
    fn empty(path: PathBuf) -> Self {
        Self {
            path,
            chunk_index: HashMap::new(),
            edits: Vec::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Load existing log from disk.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }

        // Read edits
        let mut edits = Vec::new();
        let mut max_id = 0u64;

        loop {
            // Try to read next edit
            let delta = match read_edit(&mut reader) {
                Ok(delta) => delta,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            max_id = max_id.max(delta.id);
            edits.push(delta);
        }

        // Build chunk index
        let mut chunk_index: HashMap<ChunkCoord, Vec<u64>> = HashMap::new();
        for delta in &edits {
            for &chunk in &delta.affected_chunks {
                chunk_index.entry(chunk).or_default().push(delta.id);
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            chunk_index,
            edits,
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Append edit to log, flush to disk.
    pub fn append(&mut self, mut delta: EditDelta) -> io::Result<u64> {
        // Assign ID if not set
        if delta.id == 0 {
            delta.id = self.next_id.fetch_add(1, Ordering::SeqCst);
        }

        let id = delta.id;

        // Update chunk index
        for &chunk in &delta.affected_chunks {
            self.chunk_index.entry(chunk).or_default().push(id);
        }

        // Add to in-memory edits
        self.edits.push(delta.clone());

        // Append to file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(file);

        // Write header if file is new
        let metadata = std::fs::metadata(&self.path)?;
        if metadata.len() == 0 {
            writer.write_all(MAGIC)?;
            writer.write_all(&VERSION.to_le_bytes())?;
        }

        write_edit(&mut writer, &delta)?;
        writer.flush()?;

        Ok(id)
    }

    /// Get edits affecting a chunk.
    pub fn edits_for_chunk(&self, coord: ChunkCoord) -> Vec<&EditDelta> {
        if let Some(ids) = self.chunk_index.get(&coord) {
            ids.iter()
                .filter_map(|&id| self.edits.iter().find(|e| e.id == id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all edits.
    pub fn all_edits(&self) -> &[EditDelta] {
        &self.edits
    }

    /// Get edit count.
    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    /// Rewrite log file without redundant edits.
    /// Removes edits superseded by later ones at the same position.
    pub fn compact(&mut self) -> io::Result<()> {
        // Build map of position -> latest edit
        let mut latest_edits: HashMap<String, &EditDelta> = HashMap::new();

        for edit in &self.edits {
            let key = match &edit.op {
                EditOp::SetVoxel { position, .. } | EditOp::ClearVoxel { position } => {
                    format!("point_{:.3}_{:.3}_{:.3}", position.x, position.y, position.z)
                }
                EditOp::FillRegion { region, .. } | EditOp::ClearRegion { region } => {
                    format!(
                        "region_{:.3}_{:.3}_{:.3}_{:.3}_{:.3}_{:.3}",
                        region.min.x, region.min.y, region.min.z,
                        region.max.x, region.max.y, region.max.z
                    )
                }
            };
            latest_edits.insert(key, edit);
        }

        // Keep only latest edits
        let mut compacted: Vec<EditDelta> = latest_edits.values()
            .map(|&e| e.clone())
            .collect();
        compacted.sort_by_key(|e| e.id);

        // Write to temporary file
        let temp_path = self.path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        for edit in &compacted {
            write_edit(&mut writer, edit)?;
        }

        writer.flush()?;
        drop(writer);

        // Replace original file
        std::fs::rename(&temp_path, &self.path)?;

        // Update in-memory state
        self.edits = compacted;
        self.rebuild_chunk_index();

        Ok(())
    }

    /// Write all edits to disk.
    pub fn save(&self) -> io::Result<()> {
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        for edit in &self.edits {
            write_edit(&mut writer, edit)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Rebuild chunk index from edits.
    fn rebuild_chunk_index(&mut self) {
        self.chunk_index.clear();
        for delta in &self.edits {
            for &chunk in &delta.affected_chunks {
                self.chunk_index.entry(chunk).or_default().push(delta.id);
            }
        }
    }
}

/// Read a single edit from the reader.
fn read_edit(reader: &mut impl Read) -> io::Result<EditDelta> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    let mut buf1 = [0u8; 1];

    // Read ID
    reader.read_exact(&mut buf8)?;
    let id = u64::from_le_bytes(buf8);

    // Read frame
    reader.read_exact(&mut buf4)?;
    let frame = u32::from_le_bytes(buf4);

    // Read op type
    reader.read_exact(&mut buf1)?;
    let op_type = buf1[0];

    let op = match op_type {
        0 => {
            // SetVoxel
            let position = read_vec3(reader)?;
            let voxel = read_voxel(reader)?;
            EditOp::SetVoxel { position, voxel }
        }
        1 => {
            // ClearVoxel
            let position = read_vec3(reader)?;
            EditOp::ClearVoxel { position }
        }
        2 => {
            // FillRegion
            let region = read_aabb(reader)?;
            let voxel = read_voxel(reader)?;
            EditOp::FillRegion { region, voxel }
        }
        3 => {
            // ClearRegion
            let region = read_aabb(reader)?;
            EditOp::ClearRegion { region }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown op type: {}", op_type),
            ));
        }
    };

    Ok(EditDelta::new(id, frame, op))
}

/// Write a single edit to the writer.
fn write_edit(writer: &mut impl Write, delta: &EditDelta) -> io::Result<()> {
    writer.write_all(&delta.id.to_le_bytes())?;
    writer.write_all(&delta.frame.to_le_bytes())?;

    match &delta.op {
        EditOp::SetVoxel { position, voxel } => {
            writer.write_all(&[0u8])?; // op_type
            write_vec3(writer, *position)?;
            write_voxel(writer, *voxel)?;
        }
        EditOp::ClearVoxel { position } => {
            writer.write_all(&[1u8])?; // op_type
            write_vec3(writer, *position)?;
        }
        EditOp::FillRegion { region, voxel } => {
            writer.write_all(&[2u8])?; // op_type
            write_aabb(writer, *region)?;
            write_voxel(writer, *voxel)?;
        }
        EditOp::ClearRegion { region } => {
            writer.write_all(&[3u8])?; // op_type
            write_aabb(writer, *region)?;
        }
    }

    Ok(())
}

fn read_vec3(reader: &mut impl Read) -> io::Result<Vec3> {
    let mut buf = [0u8; 12];
    reader.read_exact(&mut buf)?;
    Ok(Vec3::new(
        f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        f32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
    ))
}

fn write_vec3(writer: &mut impl Write, v: Vec3) -> io::Result<()> {
    writer.write_all(&v.x.to_le_bytes())?;
    writer.write_all(&v.y.to_le_bytes())?;
    writer.write_all(&v.z.to_le_bytes())?;
    Ok(())
}

fn read_aabb(reader: &mut impl Read) -> io::Result<Aabb> {
    let min = read_vec3(reader)?;
    let max = read_vec3(reader)?;
    Ok(Aabb::new(min, max))
}

fn write_aabb(writer: &mut impl Write, aabb: Aabb) -> io::Result<()> {
    write_vec3(writer, aabb.min)?;
    write_vec3(writer, aabb.max)?;
    Ok(())
}

fn read_voxel(reader: &mut impl Read) -> io::Result<Voxel> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(Voxel {
        color: u16::from_le_bytes([buf[0], buf[1]]),
        material_id: buf[2],
        flags: buf[3],
    })
}

fn write_voxel(writer: &mut impl Write, voxel: Voxel) -> io::Result<()> {
    writer.write_all(&voxel.color.to_le_bytes())?;
    writer.write_all(&[voxel.material_id])?;
    writer.write_all(&[voxel.flags])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("rktri_test_{}", name))
    }

    #[test]
    fn test_create_and_append() {
        let path = temp_path("create_append.rked");
        let _ = std::fs::remove_file(&path); // Clean up if exists

        let mut log = EditLog::new(&path);
        assert_eq!(log.edit_count(), 0);

        let delta = EditDelta::new(
            0,
            1,
            EditOp::SetVoxel {
                position: Vec3::new(1.0, 2.0, 3.0),
                voxel: Voxel::from_rgb565(0x1234, 5),
            },
        );

        let id = log.append(delta).unwrap();
        assert_eq!(id, 1);
        assert_eq!(log.edit_count(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_from_disk() {
        let path = temp_path("load_disk.rked");
        let _ = std::fs::remove_file(&path);

        // Create and save
        {
            let mut log = EditLog::new(&path);
            log.append(EditDelta::new(
                0,
                1,
                EditOp::SetVoxel {
                    position: Vec3::new(5.0, 5.0, 5.0),
                    voxel: Voxel::from_rgb565(0xABCD, 3),
                },
            )).unwrap();
            log.append(EditDelta::new(
                0,
                2,
                EditOp::ClearVoxel {
                    position: Vec3::new(10.0, 10.0, 10.0),
                },
            )).unwrap();
        }

        // Load and verify
        {
            let log = EditLog::load(&path).unwrap();
            assert_eq!(log.edit_count(), 2);
            assert_eq!(log.all_edits()[0].frame, 1);
            assert_eq!(log.all_edits()[1].frame, 2);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_edits_for_chunk() {
        let path = temp_path("chunk_query.rked");
        let _ = std::fs::remove_file(&path);

        let mut log = EditLog::new(&path);

        // Add edit in chunk (1, 1, 1) - position 5.0,5.0,5.0
        log.append(EditDelta::new(
            0,
            1,
            EditOp::SetVoxel {
                position: Vec3::new(5.0, 5.0, 5.0),
                voxel: Voxel::from_rgb565(0x1111, 1),
            },
        )).unwrap();

        // Add edit in chunk (0, 0, 0) - position 1.0,1.0,1.0
        log.append(EditDelta::new(
            0,
            2,
            EditOp::SetVoxel {
                position: Vec3::new(1.0, 1.0, 1.0),
                voxel: Voxel::from_rgb565(0x2222, 2),
            },
        )).unwrap();

        let chunk_1_1_1 = ChunkCoord::new(1, 1, 1);
        let edits = log.edits_for_chunk(chunk_1_1_1);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].frame, 1);

        let chunk_0_0_0 = ChunkCoord::new(0, 0, 0);
        let edits = log.edits_for_chunk(chunk_0_0_0);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].frame, 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compact() {
        let path = temp_path("compact.rked");
        let _ = std::fs::remove_file(&path);

        let mut log = EditLog::new(&path);

        let pos = Vec3::new(5.0, 5.0, 5.0);

        // Add multiple edits to same position
        log.append(EditDelta::new(
            0, 1,
            EditOp::SetVoxel {
                position: pos,
                voxel: Voxel::from_rgb565(0x1111, 1),
            },
        )).unwrap();

        log.append(EditDelta::new(
            0, 2,
            EditOp::SetVoxel {
                position: pos,
                voxel: Voxel::from_rgb565(0x2222, 2),
            },
        )).unwrap();

        log.append(EditDelta::new(
            0, 3,
            EditOp::SetVoxel {
                position: pos,
                voxel: Voxel::from_rgb565(0x3333, 3),
            },
        )).unwrap();

        assert_eq!(log.edit_count(), 3);

        log.compact().unwrap();

        // Should only keep the latest edit
        assert_eq!(log.edit_count(), 1);
        assert_eq!(log.all_edits()[0].frame, 3);

        let _ = std::fs::remove_file(&path);
    }
}
