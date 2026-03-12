use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{DynamicImage, ImageReader};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn load_image(data: &[u8]) -> PyResult<DynamicImage> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| PyValueError::new_err(format!("Cannot read image: {e}")))?;
    reader
        .decode()
        .map_err(|e| PyValueError::new_err(format!("Cannot decode image: {e}")))
}

/// Compress an image.
///
/// * `data` – raw image bytes (any supported format).
/// * `quality` – JPEG quality 1-100 (default 85). Ignored for PNG output.
/// * `max_width` / `max_height` – if set, the image is scaled down (aspect-preserved).
/// * `force_format` – `"jpeg"`, `"png"`, or `None` (auto-detect best).
///
/// Returns the compressed image bytes.
#[pyfunction]
#[pyo3(signature = (data, quality=85, max_width=None, max_height=None, force_format=None))]
pub fn compress_image(
    data: &[u8],
    quality: u8,
    max_width: Option<u32>,
    max_height: Option<u32>,
    force_format: Option<&str>,
) -> PyResult<Vec<u8>> {
    let mut img = load_image(data)?;

    // Resize if requested
    if let (Some(mw), Some(mh)) = (max_width, max_height) {
        if img.width() > mw || img.height() > mh {
            img = img.resize(mw, mh, image::imageops::FilterType::Lanczos3);
        }
    } else if let Some(mw) = max_width {
        if img.width() > mw {
            let ratio = mw as f64 / img.width() as f64;
            let new_h = (img.height() as f64 * ratio) as u32;
            img = img.resize(mw, new_h, image::imageops::FilterType::Lanczos3);
        }
    } else if let Some(mh) = max_height {
        if img.height() > mh {
            let ratio = mh as f64 / img.height() as f64;
            let new_w = (img.width() as f64 * ratio) as u32;
            img = img.resize(new_w, mh, image::imageops::FilterType::Lanczos3);
        }
    }

    // Decide output format
    let has_alpha = img.color().has_alpha();
    let fmt = match force_format {
        Some("jpeg") | Some("jpg") => "jpeg",
        Some("png") => "png",
        Some(other) => return Err(PyValueError::new_err(format!("Unsupported format: {other}"))),
        None => {
            if has_alpha {
                "png"
            } else {
                "jpeg"
            }
        }
    };

    let mut buf = Vec::new();
    match fmt {
        "jpeg" => {
            let rgb = img.to_rgb8();
            let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
            rgb.write_with_encoder(encoder)
                .map_err(|e| PyValueError::new_err(format!("JPEG encode error: {e}")))?;
        }
        "png" => {
            let rgba = img.to_rgba8();
            let encoder = PngEncoder::new(&mut buf);
            rgba.write_with_encoder(encoder)
                .map_err(|e| PyValueError::new_err(format!("PNG encode error: {e}")))?;
        }
        _ => unreachable!(),
    }

    Ok(buf)
}

/// Compress image data to JPEG with the given quality.
#[pyfunction]
#[pyo3(signature = (data, quality=85, max_width=None, max_height=None))]
pub fn compress_image_to_jpeg(
    data: &[u8],
    quality: u8,
    max_width: Option<u32>,
    max_height: Option<u32>,
) -> PyResult<Vec<u8>> {
    compress_image(data, quality, max_width, max_height, Some("jpeg"))
}

/// Compress image data to PNG.
#[pyfunction]
#[pyo3(signature = (data, max_width=None, max_height=None))]
pub fn compress_image_to_png(
    data: &[u8],
    max_width: Option<u32>,
    max_height: Option<u32>,
) -> PyResult<Vec<u8>> {
    compress_image(data, 85, max_width, max_height, Some("png"))
}
