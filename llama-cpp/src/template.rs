use std::ffi::{CString, NulError};
use std::ptr;

use crate::resolve_enable_thinking;
use llama_cpp_ffi::{llama_chat_apply_template_with_kwargs, llama_chat_message, llama_model};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[derive(Debug, Clone)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking_default: bool,
    pub enable_thinking: Option<bool>,
    pub chat_template_kwargs: Map<String, Value>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking_default: true,
            enable_thinking: None,
            chat_template_kwargs: Map::new(),
        }
    }
}

#[derive(Debug)]
pub enum ChatTemplateError {
    InteriorNul(NulError),
    OutputTooLarge,
    BridgeFailed,
    NonUtf8,
}

impl From<NulError> for ChatTemplateError {
    fn from(err: NulError) -> Self {
        Self::InteriorNul(err)
    }
}

pub fn apply_chat_template_with_kwargs(
    model: *const llama_model,
    template_override: Option<&str>,
    messages: &[ChatMessage<'_>],
    options: &ChatTemplateOptions,
) -> Result<String, ChatTemplateError> {
    let mut role_cstrings = Vec::with_capacity(messages.len());
    let mut content_cstrings = Vec::with_capacity(messages.len());

    for message in messages {
        role_cstrings.push(CString::new(message.role)?);
        content_cstrings.push(CString::new(message.content)?);
    }

    let ffi_messages: Vec<llama_chat_message> = role_cstrings
        .iter()
        .zip(content_cstrings.iter())
        .map(|(role, content)| llama_chat_message {
            role: role.as_ptr(),
            content: content.as_ptr(),
        })
        .collect();

    let template_override = template_override.map(CString::new).transpose()?;

    let mut kwargs = options.chat_template_kwargs.clone();
    if let Some(enable_thinking) = options.enable_thinking {
        kwargs.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    }

    let kwargs_json = if kwargs.is_empty() {
        None
    } else {
        Some(CString::new(Value::Object(kwargs).to_string())?)
    };

    let template_ptr = template_override
        .as_ref()
        .map_or(ptr::null(), |value| value.as_ptr());
    let kwargs_ptr = kwargs_json
        .as_ref()
        .map_or(ptr::null(), |value| value.as_ptr());
    let chat_ptr = if ffi_messages.is_empty() {
        ptr::null()
    } else {
        ffi_messages.as_ptr()
    };
    let n_msg = ffi_messages.len();

    let enable_thinking =
        resolve_enable_thinking(options.enable_thinking, options.enable_thinking_default);

    let needed = unsafe {
        llama_chat_apply_template_with_kwargs(
            model,
            template_ptr,
            chat_ptr,
            n_msg,
            options.add_generation_prompt,
            kwargs_ptr,
            enable_thinking,
            ptr::null_mut(),
            0,
        )
    };

    if needed < 0 {
        return Err(ChatTemplateError::BridgeFailed);
    }

    let needed = usize::try_from(needed).map_err(|_| ChatTemplateError::OutputTooLarge)?;
    let mut buffer = vec![0_u8; needed.saturating_add(1)];
    let length = i32::try_from(buffer.len()).map_err(|_| ChatTemplateError::OutputTooLarge)?;

    let written = unsafe {
        llama_chat_apply_template_with_kwargs(
            model,
            template_ptr,
            chat_ptr,
            n_msg,
            options.add_generation_prompt,
            kwargs_ptr,
            enable_thinking,
            buffer.as_mut_ptr().cast(),
            length,
        )
    };

    if written < 0 {
        return Err(ChatTemplateError::BridgeFailed);
    }

    let written = usize::try_from(written).map_err(|_| ChatTemplateError::OutputTooLarge)?;
    buffer.truncate(written);

    String::from_utf8(buffer).map_err(|_| ChatTemplateError::NonUtf8)
}
