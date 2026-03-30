use std::ffi::{CString, NulError};

use crate::{
    llama_context_params, llama_flash_attn_type, llama_max_devices,
    llama_max_tensor_buft_overrides, llama_model_params, llama_model_tensor_buft_override,
    llama_params_fit, mtmd_context_params,
};

const MIB_BYTES: usize = 1024 * 1024;
const DEFAULT_FIT_MARGIN_MIB: usize = 1024;

pub const FLASH_ATTN_TYPE_DISABLED: llama_flash_attn_type = 0;
pub const FLASH_ATTN_TYPE_ENABLED: llama_flash_attn_type = 1;

#[derive(Debug)]
pub enum FitError {
    InteriorNul(NulError),
    TensorSplitBufferTooSmall { required: usize, provided: usize },
    TensorBuftOverrideBufferTooSmall { required: usize, provided: usize },
}

impl From<NulError> for FitError {
    fn from(err: NulError) -> Self {
        Self::InteriorNul(err)
    }
}

pub fn flash_attn_type_from_bool(flash_attention: bool) -> llama_flash_attn_type {
    if flash_attention {
        FLASH_ATTN_TYPE_ENABLED
    } else {
        FLASH_ATTN_TYPE_DISABLED
    }
}

pub fn apply_flash_attention_alias(cparams: &mut llama_context_params, flash_attention: bool) {
    cparams.flash_attn_type = flash_attn_type_from_bool(flash_attention);
}

pub fn apply_mmproj_offload(params: &mut mtmd_context_params, mmproj_offload: bool) {
    params.use_gpu = mmproj_offload;
}

pub fn resolve_enable_thinking(
    enable_thinking: Option<bool>,
    default_enable_thinking: bool,
) -> bool {
    enable_thinking.unwrap_or(default_enable_thinking)
}

pub fn fit_required_tensor_split_len() -> usize {
    unsafe { llama_max_devices() }
}

pub fn fit_required_tensor_buft_overrides_len() -> usize {
    unsafe { llama_max_tensor_buft_overrides() }
}

pub fn fit_margins_from_mib(target_mib: &[usize], n_devices: usize) -> Vec<usize> {
    if n_devices == 0 {
        return Vec::new();
    }

    let default_margin = DEFAULT_FIT_MARGIN_MIB.saturating_mul(MIB_BYTES);
    if target_mib.is_empty() {
        return vec![default_margin; n_devices];
    }

    let mut out = vec![default_margin; n_devices];
    let mut last = target_mib[0].saturating_mul(MIB_BYTES);
    for (idx, margin) in out.iter_mut().enumerate() {
        if idx < target_mib.len() {
            last = target_mib[idx].saturating_mul(MIB_BYTES);
        }
        *margin = last;
    }
    out
}

pub fn maybe_fit_params(
    model_path: &str,
    enabled: bool,
    mparams: &mut llama_model_params,
    cparams: &mut llama_context_params,
    target_mib: &[usize],
    min_ctx: u32,
    tensor_split: &mut [f32],
    tensor_buft_overrides: &mut [llama_model_tensor_buft_override],
) -> Result<Option<i32>, FitError> {
    if !enabled {
        return Ok(None);
    }

    let required_split = fit_required_tensor_split_len();
    if tensor_split.len() < required_split {
        return Err(FitError::TensorSplitBufferTooSmall {
            required: required_split,
            provided: tensor_split.len(),
        });
    }

    let required_overrides = fit_required_tensor_buft_overrides_len();
    if tensor_buft_overrides.len() < required_overrides {
        return Err(FitError::TensorBuftOverrideBufferTooSmall {
            required: required_overrides,
            provided: tensor_buft_overrides.len(),
        });
    }

    let c_path = CString::new(model_path)?;
    let mut margins = fit_margins_from_mib(target_mib, required_split);

    let status = unsafe {
        // Keep INFO-level log output by default (ggml log level: 2).
        llama_params_fit(
            c_path.as_ptr(),
            mparams as *mut _,
            cparams as *mut _,
            tensor_split.as_mut_ptr(),
            tensor_buft_overrides.as_mut_ptr(),
            margins.as_mut_ptr(),
            min_ctx,
            2 as _,
        )
    };

    Ok(Some(status as i32))
}

#[cfg(test)]
mod tests {
    use super::{
        fit_margins_from_mib, flash_attn_type_from_bool, resolve_enable_thinking,
        FLASH_ATTN_TYPE_DISABLED, FLASH_ATTN_TYPE_ENABLED,
    };

    #[test]
    fn maps_flash_attention_alias() {
        assert_eq!(flash_attn_type_from_bool(false), FLASH_ATTN_TYPE_DISABLED);
        assert_eq!(flash_attn_type_from_bool(true), FLASH_ATTN_TYPE_ENABLED);
    }

    #[test]
    fn resolves_enable_thinking_from_request_or_default() {
        assert!(resolve_enable_thinking(Some(true), false));
        assert!(!resolve_enable_thinking(Some(false), true));
        assert!(resolve_enable_thinking(None, true));
        assert!(!resolve_enable_thinking(None, false));
    }

    #[test]
    fn fit_margins_defaults_and_expands() {
        assert_eq!(fit_margins_from_mib(&[], 0), Vec::<usize>::new());

        let one_gib = 1024usize * 1024 * 1024;
        assert_eq!(fit_margins_from_mib(&[], 2), vec![one_gib, one_gib]);

        assert_eq!(
            fit_margins_from_mib(&[512, 768], 4),
            vec![
                512usize * 1024 * 1024,
                768usize * 1024 * 1024,
                768usize * 1024 * 1024,
                768usize * 1024 * 1024,
            ]
        );
    }
}
