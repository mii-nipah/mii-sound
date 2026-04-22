//! Generic lazy-loaded resource holder with idle eviction.
//!
//! The held value is wrapped in a `std::sync::Mutex` to guarantee `Sync` even
//! when `T` itself is only `Send` (some upstream types — including
//! `voxcpm_rs::VoxCPM` — contain `OnceCell` and are not `Sync`).

use anyhow::Result;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;

pub type Held<T> = Arc<StdMutex<T>>;

struct Loaded<T> {
    value: Held<T>,
    last_used: Instant,
}

pub struct ResourceHolder<T: Send + 'static> {
    inner: Arc<AsyncMutex<Option<Loaded<T>>>>,
    ttl: Duration,
    name: Arc<str>,
    loader: Arc<dyn Fn() -> Result<T> + Send + Sync>,
}

impl<T: Send + 'static> ResourceHolder<T> {
    pub fn new<F>(name: impl Into<Arc<str>>, ttl: Duration, loader: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let holder = Self {
            inner: Arc::new(AsyncMutex::new(None)),
            ttl,
            name: name.into(),
            loader: Arc::new(loader),
        };
        holder.spawn_eviction_task();
        holder
    }

    /// Return a handle to the loaded resource, loading it on first use.
    /// Refreshes the last-used timestamp.
    pub async fn get(&self) -> Result<Held<T>> {
        {
            let mut guard = self.inner.lock().await;
            if let Some(loaded) = guard.as_mut() {
                loaded.last_used = Instant::now();
                return Ok(loaded.value.clone());
            }
        }
        log::info!("loading {}…", self.name);
        let started = Instant::now();
        let loader = self.loader.clone();
        let value = tokio::task::spawn_blocking(move || loader())
            .await
            .map_err(|e| anyhow::anyhow!("loader task panicked: {e}"))??;
        let value: Held<T> = Arc::new(StdMutex::new(value));
        let mut guard = self.inner.lock().await;
        if let Some(loaded) = guard.as_mut() {
            loaded.last_used = Instant::now();
            return Ok(loaded.value.clone());
        }
        *guard = Some(Loaded {
            value: value.clone(),
            last_used: Instant::now(),
        });
        log::info!("{} loaded in {:.2?}", self.name, started.elapsed());
        Ok(value)
    }

    fn spawn_eviction_task(&self) {
        if self.ttl.is_zero() {
            return;
        }
        let inner = self.inner.clone();
        let ttl = self.ttl;
        let name = self.name.clone();
        let tick = ttl.min(Duration::from_secs(10)).max(Duration::from_secs(1));
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.tick().await; // immediate first tick — discard
            loop {
                interval.tick().await;
                let mut guard = inner.lock().await;
                let evict = match guard.as_ref() {
                    Some(loaded) => {
                        loaded.last_used.elapsed() >= ttl
                            && Arc::strong_count(&loaded.value) == 1
                    }
                    None => false,
                };
                if evict {
                    log::info!("unloading {} (idle for {:?})", name, ttl);
                    *guard = None;
                }
            }
        });
    }
}
