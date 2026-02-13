#[allow(non_camel_case_types, non_upper_case_globals, non_snake_case, dead_code)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;
