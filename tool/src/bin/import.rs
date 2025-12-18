//! Converts Python-side assets into the packed runtime format.
//! Placeholder implementation; hook up to reference exporter in Phase 2.

use std::path::PathBuf;

fn main() {
    let output = PathBuf::from("assets/anny_body.bin");
    println!(
        "import tool placeholder — expected to write packed data to {:?}",
        output
    );
}
