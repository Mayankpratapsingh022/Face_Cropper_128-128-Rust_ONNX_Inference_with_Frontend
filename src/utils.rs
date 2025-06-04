// src/utils.rs

use std::path::PathBuf;

/// Returns `true` if the file has an image extension we support.
pub fn is_image_file(path: &PathBuf) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let ext_lc = ext.to_lowercase();
        ext_lc == "jpg"
            || ext_lc == "jpeg"
            || ext_lc == "png"
            || ext_lc == "bmp"
    } else {
        false
    }
}
