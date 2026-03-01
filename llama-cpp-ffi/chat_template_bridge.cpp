#include "chat_template_bridge.h"

#include "vendor/llama.cpp/common/chat.h"

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>

int32_t llama_chat_apply_template_with_kwargs(
    const struct llama_model * model,
    const char * tmpl_override,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    const char * chat_template_kwargs_json,
    bool enable_thinking,
    char * buf,
    int32_t length) {
    try {
        if (chat == nullptr && n_msg > 0) {
            return -1;
        }

        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_ass;
        inputs.use_jinja = true;
        inputs.enable_thinking = enable_thinking;
        inputs.messages.reserve(n_msg);

        for (size_t i = 0; i < n_msg; ++i) {
            common_chat_msg msg;
            msg.role = chat[i].role ? chat[i].role : "";
            msg.content = chat[i].content ? chat[i].content : "";
            inputs.messages.push_back(std::move(msg));
        }

        if (chat_template_kwargs_json != nullptr && chat_template_kwargs_json[0] != '\0') {
            auto kwargs_json = nlohmann::ordered_json::parse(chat_template_kwargs_json, nullptr, false);
            if (kwargs_json.is_discarded() || !kwargs_json.is_object()) {
                return -1;
            }

            for (const auto & item : kwargs_json.items()) {
                if (item.key() == "enable_thinking") {
                    if (item.value().is_boolean()) {
                        inputs.enable_thinking = item.value().template get<bool>();
                    } else if (item.value().is_string()) {
                        return -1;
                    }
                }
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        const std::string tmpl = tmpl_override ? tmpl_override : "";
        auto templates = common_chat_templates_init(model, tmpl);
        if (!templates) {
            return -1;
        }

        const auto params = common_chat_templates_apply(templates.get(), inputs);
        const std::string & prompt = params.prompt;
        const int32_t needed = static_cast<int32_t>(prompt.size());

        if (buf == nullptr || length <= 0) {
            return needed;
        }

        const int32_t n_copy = std::min<int32_t>(needed, length);
        if (n_copy > 0) {
            std::memcpy(buf, prompt.data(), static_cast<size_t>(n_copy));
        }
        if (n_copy < length) {
            buf[n_copy] = '\0';
        }

        return needed;
    } catch (...) {
        return -1;
    }
}
