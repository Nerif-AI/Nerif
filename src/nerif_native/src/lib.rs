use pyo3::prelude::*;

mod base64_ops;
mod image_compress;
mod media_utils;

#[pymodule]
fn nerif_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Image compression
    m.add_function(wrap_pyfunction!(image_compress::compress_image, m)?)?;
    m.add_function(wrap_pyfunction!(image_compress::compress_image_to_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(image_compress::compress_image_to_png, m)?)?;

    // Base64 operations
    m.add_function(wrap_pyfunction!(base64_ops::encode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::decode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::to_data_url, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::from_data_url, m)?)?;

    // Media utilities
    m.add_function(wrap_pyfunction!(media_utils::detect_image_format, m)?)?;
    m.add_function(wrap_pyfunction!(media_utils::detect_mime_type, m)?)?;
    m.add_function(wrap_pyfunction!(media_utils::resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(media_utils::has_transparency, m)?)?;
    m.add_function(wrap_pyfunction!(media_utils::get_image_dimensions, m)?)?;

    Ok(())
}
