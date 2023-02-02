#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod eval;
mod parse;

use parse::{total_action, Action};
use std::collections::HashMap;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn evaluate_equation(equation: &str) -> String {
    let parsed_result: Result<Action, pom::Error> = total_action().parse(equation.as_bytes());
    let context = HashMap::new();

    match parsed_result {
        Ok(action) => match action {
            Action::DefineFunc(fname, vars, f) => fname,
            Action::DefineVar(var, token) => var.to_string(),
            Action::EvalExpr(f) => match f.eval(context) {
                Ok(expr) => expr.to_string(),
                Err(_err) => "nope".to_string(),
            },
            _ => "adf".to_string(),
        },
        Err(_err) => {
            panic!("nope")
        }
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![evaluate_equation])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
