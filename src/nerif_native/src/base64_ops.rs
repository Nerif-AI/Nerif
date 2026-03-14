use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Encode raw bytes to a base64 string.
#[pyfunction]
pub fn encode_base64(data: &[u8]) -> String {
    STANDARD.encode(data)
}

/// Decode a base64 string to raw bytes.
#[pyfunction]
pub fn decode_base64(data: &str) -> PyResult<Vec<u8>> {
    STANDARD
        .decode(data)
        .map_err(|e| PyValueError::new_err(format!("Invalid base64: {e}")))
}

/// Build a data URL: `data:<mime>;base64,<encoded>`.
#[pyfunction]
pub fn to_data_url(data: &[u8], mime_type: &str) -> String {
    let encoded = STANDARD.encode(data);
    format!("data:{mime_type};base64,{encoded}")
}

/// Parse a data URL and return `(mime_type, raw_bytes)`.
#[pyfunction]
pub fn from_data_url(url: &str) -> PyResult<(String, Vec<u8>)> {
    let url = url.trim();
    if !url.starts_with("data:") {
        return Err(PyValueError::new_err("Not a data URL"));
    }
    let rest = &url[5..];
    let (header, b64) = rest
        .split_once(",")
        .ok_or_else(|| PyValueError::new_err("Malformed data URL: missing comma"))?;

    let mime = header
        .split(';')
        .next()
        .unwrap_or("application/octet-stream")
        .to_string();

    let bytes = STANDARD
        .decode(b64)
        .map_err(|e| PyValueError::new_err(format!("Invalid base64 in data URL: {e}")))?;

    Ok((mime, bytes))
}
