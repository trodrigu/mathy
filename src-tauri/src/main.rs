#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod parse;
mod eval;

use parse::{total_expr, Token};

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn evaluate_equation(equation: &str) -> String {
    let result: Result<Token, pom::Error> = total_expr().parse(equation.as_bytes());
    match result {
        //Ok(token) => eval(token, 2.0).to_string(),
        Ok(token) => "hi".to_string(),
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
