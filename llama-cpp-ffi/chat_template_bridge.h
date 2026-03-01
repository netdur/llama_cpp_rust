#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t llama_chat_apply_template_with_kwargs(
    const struct llama_model * model,
    const char * tmpl_override,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    const char * chat_template_kwargs_json,
    bool enable_thinking,
    char * buf,
    int32_t length);

#ifdef __cplusplus
}
#endif
