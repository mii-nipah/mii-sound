//! Backend-agnostic VoxCPM model loading + synthesis. Used by the worker
//! process. Not used directly by the frontend (`serve`), which talks to the
//! worker over pipes.

use crate::proto::TtsRequest;
use anyhow::{Result, anyhow};
use bytes::Bytes;
use std::path::Path;
use voxcpm_rs::{
    CancelToken, Error as VoxError, GenerateOptions, Prompt, PromptAudio, VoxCPM, audio,
};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};

type CpuBackend = NdArray<f32, i32>;
type GpuBackend = Wgpu<f32, i32>;

pub enum Model {
    Cpu(VoxCPM<CpuBackend>),
    Gpu(VoxCPM<GpuBackend>),
}

pub fn load(model_dir: &Path, cpu: bool) -> Result<Model> {
    if cpu {
        let device = NdArrayDevice::default();
        let m = VoxCPM::<CpuBackend>::from_local(model_dir, &device)
            .map_err(|e| anyhow!("failed to load tts model from {}: {e}", model_dir.display()))?;
        Ok(Model::Cpu(m))
    } else {
        let device = WgpuDevice::default();
        let m = VoxCPM::<GpuBackend>::from_local(model_dir, &device)
            .map_err(|e| anyhow!("failed to load tts model from {}: {e}", model_dir.display()))?;
        Ok(Model::Gpu(m))
    }
}

pub enum SynthError {
    Cancelled,
    Other(String),
}

pub fn synthesize(
    model: &Model,
    req: TtsRequest,
    inline_audio: Option<Bytes>,
    cancel: CancelToken,
) -> Result<Bytes, SynthError> {
    match model {
        Model::Cpu(m) => synth_inner(m, req, inline_audio, cancel),
        Model::Gpu(m) => synth_inner(m, req, inline_audio, cancel),
    }
}

fn synth_inner<B: burn::tensor::backend::Backend>(
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
    use std::path::PathBuf;
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
