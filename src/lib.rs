//! matplotlib-rs: a Rust reimplementation of matplotlib's plotting pipeline.

pub mod agg;
pub mod backend;
pub mod font;
pub mod mpl;
pub mod pool;
pub mod python;

/// The allocator of the Rust core.  Drawing allocates and frees large
/// buffers (the vertices of long lines, scanlines, images) again and again:
/// mimalloc keeps them for the next use, where the system allocator maps and
/// unmaps their memory every time.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
