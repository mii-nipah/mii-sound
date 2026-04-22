//! Transport-level helpers (socket name resolution, env helpers).

use anyhow::{Result, anyhow};
use interprocess::local_socket::{
    GenericFilePath, GenericNamespaced, Name, NameType, ToFsName, ToNsName,
};
use std::path::{Path, PathBuf};

/// Default *filesystem* socket path for unix-like systems. On Windows we still
/// expose this as a fallback, though `default_name()` prefers the namespaced
/// form there.
pub fn default_socket_path() -> PathBuf {
    if let Some(rt) = dirs::runtime_dir() {
        return rt.join("mii-sound.sock");
    }
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn getuid() -> u32;
        }
        let uid = unsafe { getuid() };
        return PathBuf::from(format!("/tmp/mii-sound-{uid}.sock"));
    }
    #[cfg(not(unix))]
    {
        PathBuf::from("mii-sound.sock")
    }
}

/// Resolve the local-socket [`Name`] used by both client and server.
///
/// - If the user passed `--socket <path>`, treat it as a filesystem path.
/// - Otherwise on platforms that support a namespace, use the namespaced name
///   `mii-sound.sock` (Windows, abstract sockets on Linux); otherwise fall back
///   to a filesystem path under `$XDG_RUNTIME_DIR`.
pub fn resolve_name(custom: Option<&Path>) -> Result<Name<'static>> {
    if let Some(path) = custom {
        let owned = path.to_path_buf();
        let name = owned
            .clone()
            .to_fs_name::<GenericFilePath>()
            .map_err(|e| anyhow!("invalid socket path {}: {e}", owned.display()))?;
        return Ok(name);
    }
    if GenericNamespaced::is_supported() {
        return "mii-sound.sock"
            .to_ns_name::<GenericNamespaced>()
            .map_err(|e| anyhow!("invalid namespaced socket name: {e}"));
    }
    let path = default_socket_path();
    path.clone()
        .to_fs_name::<GenericFilePath>()
        .map_err(|e| anyhow!("invalid socket path {}: {e}", path.display()))
}

pub fn token_from_env() -> Result<String> {
    std::env::var("TOKEN").map_err(|_| anyhow!("$TOKEN env var must be set for network mode"))
}
