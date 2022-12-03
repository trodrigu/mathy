#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn evaluate_equation(equation: &str) -> String {
    let result: f32 = parse_equation(equation);
    format!("x = {}", result)
}

#[derive(PartialEq, Clone, Debug)]
pub enum Token {
    Var(String),
    Plus,
    Minus,
    Division,
    Multiplication,
    Constant(f32),
    Equal,
}

fn parse_token(char: &str) -> Token {
    match char {
        "a" => Token::Var(String::from("a")),
        "b" => Token::Var(String::from("b")),
        "c" => Token::Var(String::from("c")),
        "d" => Token::Var(String::from("d")),
        "e" => Token::Var(String::from("e")),
        "f" => Token::Var(String::from("f")),
        "g" => Token::Var(String::from("g")),
        "h" => Token::Var(String::from("h")),
        "i" => Token::Var(String::from("i")),
        "j" => Token::Var(String::from("j")),
        "k" => Token::Var(String::from("k")),
        "l" => Token::Var(String::from("l")),
        "m" => Token::Var(String::from("m")),
        "n" => Token::Var(String::from("n")),
        "o" => Token::Var(String::from("o")),
        "p" => Token::Var(String::from("p")),
        "q" => Token::Var(String::from("q")),
        "r" => Token::Var(String::from("r")),
        "s" => Token::Var(String::from("s")),
        "t" => Token::Var(String::from("t")),
        "u" => Token::Var(String::from("u")),
        "v" => Token::Var(String::from("v")),
        "w" => Token::Var(String::from("w")),
        "x" => Token::Var(String::from("x")),
        "y" => Token::Var(String::from("y")),
        "z" => Token::Var(String::from("z")),
        "+" => Token::Plus,
        "-" => Token::Minus,
        "/" => Token::Division,
        "*" => Token::Multiplication,
        "=" => Token::Equal,
        _ => Token::Constant(char.parse::<f32>().unwrap()),
    }
}

fn parse_char(char: &str) -> Option<Token> {
    match char {
        " " => None,
        parsed_char => Some(parse_token(parsed_char)),
    }
}

fn parse_equation(equation: &str) -> f32 {
    let mut parsed_equation: Vec<Token> = Vec::new();

    for char in equation.chars() {
        if let Some(parsed_char) = parse_char(&(char.to_string())) {
            parsed_equation.push(parsed_char);
        }
    }

    let mut sides: Vec<Vec<Token>> = Vec::new();

    let parsed_equation_iter = parsed_equation.split(|el| el == &Token::Equal);

    for side in parsed_equation_iter {
        sides.push(side.to_vec());
    }

    let right: Vec<Token> = sides.pop().unwrap();
    let left: Vec<Token> = sides.pop().unwrap();

    let left_grouped: Vec<Vec<Token>> = group_tokens(left);

    let right_grouped: Vec<Vec<Token>> = group_tokens(right);

    // TODO: these don't allow constants on vars like 2x or 6x
    // Push the tupelize in here
    // Create something like a term where its:
    // -2x^2
    // (Operator, Coefficient, Variable, Exponent)
    // (Some(-), Some(Constant(2)), Some(Var(x)), Some(Constant(2)))
    // -2^2
    // (Some(-), Some(Constant(2)), None, Some(Constant(2)))
    // -2
    // (Some(-), Some(Constant(2)), None, None)
    // 2
    // (None, Some(Constant(2)), None, None)
    let (constants_and_operators_left, vars_left): (Vec<Vec<Token>>, Vec<Vec<Token>>) =
        left_grouped
            .into_iter()
            .partition(|el| el.iter().any(|t| matches!(t, Token::Constant(_))));

    let (constants_and_operators_right, vars_right): (Vec<Vec<Token>>, Vec<Vec<Token>>) =
        right_grouped
            .into_iter()
            .partition(|el| el.iter().any(|t| matches!(t, Token::Constant(_))));

    let mut constants_and_operators_left_tuples: Vec<(Option<Token>, Token)> =
        tupelize_side(constants_and_operators_left);
    let mut constants_and_operators_right_tuples: Vec<(Option<Token>, Token)> =
        tupelize_side(constants_and_operators_right);
    let mut vars_left_tuples: Vec<(Option<Token>, Token)> = tupelize_side(vars_left);
    let mut vars_right_tuples: Vec<(Option<Token>, Token)> = tupelize_side(vars_right);

    dbg!(constants_and_operators_left_tuples.clone());
    dbg!(constants_and_operators_right_tuples.clone());
    dbg!(vars_left_tuples.clone());
    dbg!(vars_right_tuples.clone());

    while let Some(el) = constants_and_operators_left_tuples.pop() {
        let negated_el = negate(el);
        constants_and_operators_right_tuples.push(negated_el);
    }

    while let Some(el) = vars_right_tuples.pop() {
        let negated_el = negate(el);
        vars_left_tuples.push(negated_el);
    }

    collapse_right(constants_and_operators_right_tuples)
}

fn tupelize_side(side: Vec<Vec<Token>>) -> Vec<(Option<Token>, Token)> {
    side.iter()
        .map(|el| {
            let mut el = el.clone();
            let constant = el.pop().unwrap();
            if let Some(possible_operator) = el.pop() {
                (Some(possible_operator), constant)
            } else {
                (None, constant)
            }
        })
        .collect::<Vec<(Option<Token>, Token)>>()
}

fn group_tokens(tokens: Vec<Token>) -> Vec<Vec<Token>> {
    let last_index = tokens.len() - 1;
    tokens
        .iter()
        .enumerate()
        .fold(vec![Vec::new()], |acc: Vec<Vec<Token>>, (i, el)| {
            let mut acc_cloned = acc.clone();
            match el {
                Token::Plus => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Plus);
                    acc_cloned.push(last_vector);
                    acc_cloned
                }
                Token::Minus => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Minus);
                    acc_cloned.push(last_vector);
                    acc_cloned
                }
                Token::Multiplication => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Multiplication);
                    acc_cloned.push(last_vector);
                    acc_cloned
                }
                Token::Division => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Division);
                    acc_cloned.push(last_vector);
                    acc_cloned
                }
                Token::Var(constant) => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Var(constant.to_string()));
                    acc_cloned.push(last_vector);
                    if i != last_index {
                        // Start new term
                        acc_cloned.push(Vec::new());
                    }
                    acc_cloned
                }
                Token::Constant(val) => {
                    let mut last_vector = acc_cloned.pop().unwrap();
                    last_vector.push(Token::Constant(*val));
                    acc_cloned.push(last_vector);
                    // 2x+1 = 3
                    // 1+2x = 3
                    if i != last_index {
                        acc_cloned.push(Vec::new());
                    }
                    acc_cloned
                }
                _ => acc,
            }
        })
}

fn collapse_right(right: Vec<(Option<Token>, Token)>) -> f32 {
    right.iter().fold(
        0.0,
        |acc_val, maybe_operator_and_constant| match maybe_operator_and_constant {
            (None, Token::Constant(val)) => acc_val + *val,
            (Some(Token::Plus), Token::Constant(val)) => acc_val + *val,
            (Some(Token::Minus), Token::Constant(val)) => acc_val - *val,
            (Some(Token::Multiplication), Token::Constant(val)) => acc_val * *val,
            (Some(Token::Division), Token::Constant(val)) => acc_val / *val,
            _ => acc_val,
        },
    )
}

fn negate(side_tuple: (Option<Token>, Token)) -> (Option<Token>, Token) {
    match side_tuple {
        (None, Token::Constant(val)) => (None, Token::Constant(val)),
        (None, Token::Var(val)) => (None, Token::Var(val)),
        (Some(Token::Plus), Token::Constant(val)) => (Some(Token::Minus), Token::Constant(val)),
        (Some(Token::Minus), Token::Constant(val)) => (Some(Token::Plus), Token::Constant(val)),
        (Some(Token::Multiplication), Token::Constant(val)) => {
            (Some(Token::Division), Token::Constant(val))
        }
        (Some(Token::Division), Token::Constant(val)) => {
            (Some(Token::Multiplication), Token::Constant(val))
        }
        (Some(Token::Plus), Token::Var(val)) => (Some(Token::Minus), Token::Var(val)),
        (Some(Token::Minus), Token::Var(val)) => (Some(Token::Plus), Token::Var(val)),
        (Some(Token::Multiplication), Token::Var(val)) => (Some(Token::Division), Token::Var(val)),
        (Some(Token::Division), Token::Var(val)) => (Some(Token::Multiplication), Token::Var(val)),

        (None, token) => (None, token),
        (Some(optional_token), token) => (Some(optional_token), token),
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![evaluate_equation])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
