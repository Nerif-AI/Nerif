use std::io::Cursor;

use image::ImageReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Detect the image format from raw bytes.
/// Returns a lowercase string like `"jpeg"`, `"png"`, `"gif"`, `"webp"`, or `"unknown"`.
#[pyfunction]
pub fn detect_image_format(data: &[u8]) -> String {
    let reader = match ImageReader::new(Cursor::new(data)).with_guessed_format() {
        Ok(r) => r,
        Err(_) => return "unknown".to_string(),
    };
    match reader.format() {
        Some(fmt) => format!("{fmt:?}").to_lowercase(),
        None => "unknown".to_string(),
    }
}

/// Return a MIME type string for the given image bytes.
#[pyfunction]
pub fn detect_mime_type(data: &[u8]) -> String {
    let fmt = detect_image_format(data);
    match fmt.as_str() {
        "jpeg" => "image/jpeg".to_string(),
        "png" => "image/png".to_string(),
        "gif" => "image/gif".to_string(),
        "webp" => "image/webp".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Resize image bytes to fit within `max_width` x `max_height`, preserving aspect ratio.
/// Returns the resized image bytes in the same format (or JPEG if format is unknown).
#[pyfunction]
#[pyo3(signature = (data, max_width, max_height, quality=85))]
pub fn resize_image(data: &[u8], max_width: u32, max_height: u32, quality: u8) -> PyResult<Vec<u8>> {
    crate::image_compress::compress_image(data, quality, Some(max_width), Some(max_height), None)
}

/// Check whether an image has any transparent pixels.
#[pyfunction]
pub fn has_transparency(data: &[u8]) -> PyResult<bool> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| PyValueError::new_err(format!("Cannot read image: {e}")))?;
    let img = reader
        .decode()
        .map_err(|e| PyValueError::new_err(format!("Cannot decode image: {e}")))?;

    if !img.color().has_alpha() {
        return Ok(false);
    }

    let rgba = img.to_rgba8();
    for pixel in rgba.pixels() {
        if pixel[3] < 255 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Return `(width, height)` of an image without fully decoding it.
#[pyfunction]
pub fn get_image_dimensions(data: &[u8]) -> PyResult<(u32, u32)> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| PyValueError::new_err(format!("Cannot read image: {e}")))?;
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| PyValueError::new_err(format!("Cannot get dimensions: {e}")))?;
    Ok((w, h))
}
