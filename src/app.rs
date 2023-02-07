use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;
use std::collections::HashMap;
use mathy_core::parse::Token;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "tauri"])]
    async fn invoke(cmd: &str, args: JsValue) -> JsValue;

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Serialize, Deserialize)]
struct EquationArgs<'a> {
    equation: &'a str,
}

#[derive(Clone, PartialEq, Debug)]
struct State {
    current_equation: Option<String>,
    tokens: HashMap<String, Token>,
    last_evaluated: Option<String>
}

impl State {
    pub fn default() -> State {
        Self {
            current_equation: Some("".to_string()),
            tokens: HashMap::new(),
            last_evaluated: Some("".to_string())
        }
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let equation_input_ref = use_node_ref();

    let equation = use_state(|| State::default().current_equation.unwrap());

    let equation_msg = use_state(|| State::default().last_evaluated.unwrap());
    {
        let equation_msg = equation_msg.clone();
        let equation = equation.clone();
        let updated_equation = equation.clone();
        use_effect_with_deps(
            move |_| {
                spawn_local(async move {
                    if equation.is_empty() {
                        return;
                    }

                    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
                    let new_msg = invoke(
                        "evaluate_equation",
                        to_value(&EquationArgs {
                            equation: &*equation,
                        })
                        .unwrap(),
                    )
                    .await;
                    log(&new_msg.as_string().unwrap());
                    equation_msg.set(new_msg.as_string().unwrap());
                });

                || {}
            },
            updated_equation,
        );
    }

    let evaluate_equation = {
        let equation = equation.clone();
        let equation_input_ref = equation_input_ref.clone();
        Callback::from(move |_| {
            equation.set(
                equation_input_ref
                    .cast::<web_sys::HtmlInputElement>()
                    .unwrap()
                    .value(),
            );
        })
    };

    html! {
        <main class="container">
            <div class="row">
                <input id="equation-input" ref={equation_input_ref} placeholder="Enter an equation..." />
                <button type="button" onclick={evaluate_equation}>{"Evaluate"}</button>
            </div>

            <p><b>{ &*equation_msg }</b></p>
        </main>
    }
}
