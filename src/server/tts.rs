//! Server-side TTS request handler.

use crate::proto::TtsRequest;
use crate::server::holder::{Held, ResourceHolder};
use anyhow::{Result, anyhow};
use bytes::Bytes;
use std::path::PathBuf;
use voxcpm_rs::{CancelToken, Error as VoxError, GenerateOptions, Prompt, PromptAudio, VoxCPM, audio};

use burn::backend::{NdArray, Wgpu};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;

type CpuBackend = NdArray<f32, i32>;
type GpuBackend = Wgpu<f32, i32>;

/// Backend-agnostic TTS engine wrapper. Holds whichever backend was selected
/// at server start.
pub enum TtsEngine {
    Cpu(ResourceHolder<VoxCPM<CpuBackend>>),
    Gpu(ResourceHolder<VoxCPM<GpuBackend>>),
}

impl TtsEngine {
    pub fn new(model_dir: PathBuf, ttl: std::time::Duration, cpu: bool) -> Self {
        if cpu {
            let dir = model_dir;
            TtsEngine::Cpu(ResourceHolder::new("tts model (cpu)", ttl, move || {
                let device = NdArrayDevice::default();
                VoxCPM::<CpuBackend>::from_local(&dir, &device)
                    .map_err(|e| anyhow!("failed to load tts model from {}: {e}", dir.display()))
            }))
        } else {
            let dir = model_dir;
            TtsEngine::Gpu(ResourceHolder::new("tts model (gpu)", ttl, move || {
                let device = WgpuDevice::default();
                VoxCPM::<GpuBackend>::from_local(&dir, &device)
                    .map_err(|e| anyhow!("failed to load tts model from {}: {e}", dir.display()))
            }))
        }
    }

    pub async fn generate(
        &self,
        req: TtsRequest,
        inline_audio: Option<Bytes>,
        cancel: CancelToken,
    ) -> TtsResult {
        match self {
            TtsEngine::Cpu(holder) => match holder.get().await {
                Ok(model) => run_blocking(model, req, inline_audio, cancel).await,
                Err(e) => TtsResult::ModelMissing(e.to_string()),
            },
            TtsEngine::Gpu(holder) => match holder.get().await {
                Ok(model) => run_blocking(model, req, inline_audio, cancel).await,
                Err(e) => TtsResult::ModelMissing(e.to_string()),
            },
        }
    }
}

pub enum TtsResult {
    Ok(Bytes),
    ModelMissing(String),
    Cancelled,
    Failed(String),
}

async fn run_blocking<B>(
    model: Held<VoxCPM<B>>,
    req: TtsRequest,
    inline_audio: Option<Bytes>,
    cancel: CancelToken,
) -> TtsResult
where
    B: burn::tensor::backend::Backend,
{
    let join = tokio::task::spawn_blocking(move || {
        let guard = model.lock().expect("model mutex poisoned");
        synthesize(&*guard, req, inline_audio, cancel)
    });
    match join.await {
        Ok(Ok(bytes)) => TtsResult::Ok(bytes),
        Ok(Err(SynthError::Cancelled)) => TtsResult::Cancelled,
        Ok(Err(SynthError::Other(e))) => TtsResult::Failed(e),
        Err(e) => TtsResult::Failed(format!("worker task panicked: {e}")),
    }
}

enum SynthError {
    Cancelled,
    Other(String),
}

fn synthesize<B: burn::tensor::backend::Backend>(
    model: &VoxCPM<B>,
    req: TtsRequest,
    inline_audio: Option<Bytes>,
    cancel: CancelToken,
) -> Result<Bytes, SynthError> {
    let prompt = build_prompt(&req, inline_audio).map_err(|e| SynthError::Other(e.to_string()))?;

    let mut builder = GenerateOptions::builder()
        .cfg(req.cfg)
        .timesteps(req.steps as usize)
        .cancel(cancel);
    if let Some(p) = prompt {
        builder = builder.prompt(p);
    }
    let opts = builder.build();

    let samples = match model.generate(&req.text, opts) {
        Ok(s) => s,
        Err(VoxError::Cancelled) => return Err(SynthError::Cancelled),
        Err(e) => return Err(SynthError::Other(format!("generation failed: {e}"))),
    };

    audio::encode_wav(&samples, model.sample_rate())
        .map(Bytes::from)
        .map_err(|e| SynthError::Other(format!("encoding wav failed: {e}")))
}

fn build_prompt(req: &TtsRequest, inline_audio: Option<Bytes>) -> Result<Option<Prompt>> {
    let audio: Option<PromptAudio> = if req.inline_reference {
        let bytes = inline_audio
            .ok_or_else(|| anyhow!("inline reference flagged but no audio bytes received"))?;
        Some(PromptAudio::Encoded(bytes.to_vec()))
    } else if let Some(path) = req.reference.as_ref() {
        Some(PathBuf::from(path).into())
    } else {
        None
    };
    Ok(match (audio, req.continuation.as_ref()) {
        (None, _) => None,
        (Some(a), None) => Some(Prompt::Reference { audio: a }),
        (Some(a), Some(text)) => Some(Prompt::Continuation {
            audio: a,
            text: text.clone(),
        }),
    })
}


