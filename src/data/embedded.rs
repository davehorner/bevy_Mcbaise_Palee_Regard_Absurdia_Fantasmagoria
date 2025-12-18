//! Embedded data source helper using `include_bytes!`.
//!
//! TODO: wire to packed artifacts once available.

#[derive(Debug)]
pub enum DataSource {
    Embedded(&'static [u8]),
    External(String),
}
