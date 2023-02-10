use mathy_core::parse::{
    total_action,
    Action::{DefineFunc, DefineVar, EvalExpr},
    Token,
};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::to_value;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlInputElement;
use yew::prelude::*;

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
    current_equation_or_action: Option<String>,
    tokens: HashMap<String, Token>,
    last_evaluated: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            current_equation_or_action: None,
            tokens: HashMap::new(),
            last_evaluated: None,
        }
    }
}

#[derive(Debug)]
pub enum StateAction {
    EvalOrDefine(String),
}

impl Reducible for State {
    type Action = StateAction;

    fn reduce(self: std::rc::Rc<Self>, action: Self::Action) -> std::rc::Rc<Self> {
        log(&"yo!");
        match action {
            StateAction::EvalOrDefine(equation_or_action) => {
                let mut current_tokens = self.tokens.clone();
                let action = total_action().parse(equation_or_action.as_bytes()).unwrap();

                match action {
                    DefineFunc(func_name, args, expr) => {
                        current_tokens.insert(func_name, expr);
                        Self {
                            current_equation_or_action: None,
                            tokens: current_tokens,
                            last_evaluated: Some(equation_or_action),
                        }
                        .into()
                    }
                    DefineVar(var_name, var) => {
                        current_tokens.insert(var_name, var);
                        Self {
                            current_equation_or_action: None,
                            tokens: current_tokens,
                            last_evaluated: Some(equation_or_action),
                        }
                        .into()
                    }
                    EvalExpr(expr) => {
                        let tokens = current_tokens.clone();
                        let last_evaluated =
                            Some(expr.eval(current_tokens.clone()).unwrap().to_string());
                        Self {
                            current_equation_or_action: None,
                            tokens: tokens,
                            last_evaluated: last_evaluated,
                        }
                        .into()
                    }
                }
            }
        }
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let state = use_reducer(|| State::default());

    let evaluate_equation = {
        let state = state.clone();
        Callback::from(move |equation_or_action| {
            state.dispatch(StateAction::EvalOrDefine(equation_or_action))
        })
    };

    let onkeypress = {
        move |e: KeyboardEvent| {
            if e.key() == "Enter" {
                let input: HtmlInputElement = e.target_unchecked_into();
                let value = input.value();

                input.set_value("");
                evaluate_equation.emit(value);
            }
        }
    };

    html! {
        <main class="container">
            <div class="row">
                <input id="equation-input" {onkeypress} placeholder="Enter an equation..." />
            </div>

            <p><b>
            {
                if let Some(evaluated) = &state.last_evaluated {
                    evaluated.clone()
                } else {
                    "None Evaluated Yet!".to_string()
                }
            }</b></p>
        </main>
    }
}
