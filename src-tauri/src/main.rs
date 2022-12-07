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
    Var(f32, String, i32),
    Plus,
    Minus,
    Division,
    Multiplication,
    Constant(f32),
    Equal,
    Exponent,
}

fn parse_term(maybe_term: &str) -> Token {
    if is_equal(maybe_term.to_string()) {
        Token::Equal
    } else if is_operator(maybe_term.to_string()) {
        parse_operator(maybe_term)
    } else {
        maybe_term.split('^').fold(Token::Constant(1.0), |acc, el| {
            el.chars().fold(acc, |inner_acc, inner_el| {
                let acc_cloned = inner_acc.clone();
                if is_equal(inner_el.to_string()) {
                    Token::Equal
                } else if matches!(acc_cloned, Token::Var(_, _, _)) {
                    if let Token::Var(signed_float, var_str, _int) = acc_cloned {
                        let signed_integer = inner_el.to_string().parse::<i32>().unwrap();
                        Token::Var(signed_float, var_str, signed_integer)
                    } else {
                        acc_cloned
                    }
                } else if inner_el.is_digit(10) {
                    if let Token::Constant(sign) = acc_cloned {
                        let current_float = inner_el.to_string().parse::<f32>().unwrap();
                        Token::Constant(sign * current_float)
                    } else {
                        acc_cloned
                    }
                } else if is_sign(inner_el.to_string()) {
                    let mut sign = inner_el.to_string();
                    sign.push_str("1.0");
                    let signed_float = sign.parse::<f32>().unwrap();
                    Token::Constant(signed_float)
                } else if char::is_ascii_alphabetic(&inner_el) {
                    if let Token::Constant(signed_float) = acc_cloned {
                        Token::Var(signed_float, inner_el.to_string(), 1)
                    } else {
                        acc_cloned
                    }
                } else {
                    acc_cloned
                }
            })
        })
    }
}

fn is_sign(str: String) -> bool {
    str == "-" || str == "+"
}

fn is_equal(str: String) -> bool {
    str == "="
}

fn is_operator(str: String) -> bool {
    str == "-" || str == "+" || str == "/" || str == "*"
}

fn parse_operator(str: &str) -> Token {
    match str {
        "-" => Token::Minus,
        "+" => Token::Plus,
        "/" => Token::Division,
        "*" => Token::Multiplication,
        _ => Token::Constant(1.0),
    }
}

fn parse_equation(equation: &str) -> f32 {
    let mut parsed_equation: Vec<Token> = Vec::new();

    for maybe_term in equation.split(' ') {
        let term = parse_term(maybe_term);
        parsed_equation.push(term);
    }

    let mut sides: Vec<Vec<Token>> = Vec::new();

    let parsed_equation_iter = parsed_equation.split(|el| el == &Token::Equal);

    for side in parsed_equation_iter {
        sides.push(side.to_vec());
    }

    let right: Vec<Token> = sides.pop().unwrap();

    let left: Vec<Token> = sides.pop().unwrap();

    dbg!(right);
    dbg!(left);

    //let new_left: Vec<Term> = Vec::new();

    //for left_el in left_grouped.iter() {
    //if matches!(left_el, Token::Var) {
    //new_left.push(left_el);
    //} else {
    //right_grouped.push(left_el);
    //}
    //}

    //while let Some(el) = constants_and_operators_left_tuples.pop() {
    //let negated_el = negate(el);
    //constants_and_operators_right_tuples.push(negated_el);
    //}

    //while let Some(el) = vars_right_tuples.pop() {
    //let negated_el = negate(el);
    //vars_left_tuples.push(negated_el);
    //}

    //collapse_right(constants_and_operators_right_tuples)
    0.0
}

//fn collapse_right(right: Vec<(Option<Token>, Token)>) -> f32 {
//right.iter().fold(
//0.0,
//|acc_val, maybe_operator_and_constant| match maybe_operator_and_constant {
//(None, Token::Constant(val)) => acc_val + *val,
//(Some(Token::Plus), Token::Constant(val)) => acc_val + *val,
//(Some(Token::Minus), Token::Constant(val)) => acc_val - *val,
//(Some(Token::Multiplication), Token::Constant(val)) => acc_val * *val,
//(Some(Token::Division), Token::Constant(val)) => acc_val / *val,
//_ => acc_val,
//},
//)
//}

//fn negate(side_tuple: (Option<Token>, Token)) -> (Option<Token>, Token) {
//match side_tuple {
//(None, Token::Constant(val)) => (None, Token::Constant(val)),
//(None, Token::Var(val)) => (None, Token::Var(val)),
//(Some(Token::Plus), Token::Constant(val)) => (Some(Token::Minus), Token::Constant(val)),
//(Some(Token::Minus), Token::Constant(val)) => (Some(Token::Plus), Token::Constant(val)),
//(Some(Token::Multiplication), Token::Constant(val)) => {
//(Some(Token::Division), Token::Constant(val))
//}
//(Some(Token::Division), Token::Constant(val)) => {
//(Some(Token::Multiplication), Token::Constant(val))
//}
//(Some(Token::Plus), Token::Var(val)) => (Some(Token::Minus), Token::Var(val)),
//(Some(Token::Minus), Token::Var(val)) => (Some(Token::Plus), Token::Var(val)),
//(Some(Token::Multiplication), Token::Var(val)) => (Some(Token::Division), Token::Var(val)),
//(Some(Token::Division), Token::Var(val)) => (Some(Token::Multiplication), Token::Var(val)),

//(None, token) => (None, token),
//(Some(optional_token), token) => (Some(optional_token), token),
//}
//}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![evaluate_equation])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
