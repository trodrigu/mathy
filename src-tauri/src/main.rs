#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use pom::parser::Parser;
use pom::parser::*;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn evaluate_equation(equation: &str) -> String {
    let result: Result<Vec<Token>, pom::Error> = expression().parse(equation.as_bytes());
    dbg!(result);
    "".to_string()
}

#[derive(PartialEq, Clone, Debug)]
pub enum Token {
    Var(String),
    Op(Operator),
    Constant(f32),
    Equal,
    RightParen,
    LeftParen,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Exponent,
    Modulo,
}

fn variable<'a>() -> Parser<'a, u8, Token> {
    one_of(b"abcdefghijklmnopqrstuvwxyz")
        .repeat(1..)
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
        .map(Token::Var)
}
fn right_paren<'a>() -> Parser<'a, u8, Token> {
    sym(b')').map(|_sy| Token::RightParen)
}
fn left_paren<'a>() -> Parser<'a, u8, Token> {
    sym(b'(').map(|_sy| Token::LeftParen)
}
fn equal<'a>() -> Parser<'a, u8, Token> {
    sym(b'=').map(|_sy| Token::Equal)
}
fn number<'a>() -> Parser<'a, u8, Token> {
    let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
    let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
    let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
    let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
    number
        .collect()
        .convert(|v| String::from_utf8(v.to_vec()))
        .convert(|s| s.parse::<f32>())
        .map(Token::Constant)
}

fn operator<'a>() -> Parser<'a, u8, Token> {
    one_of(b"+-*/^%")
        .map(|sy| match sy {
            b'+' => Operator::Add,
            b'-' => Operator::Subtract,
            b'*' => Operator::Multiply,
            b'/' => Operator::Divide,
            b'^' => Operator::Exponent,
            b'%' => Operator::Modulo,
            _ => Operator::Add,
        })
        .map(Token::Op)
}

fn expression<'a>() -> Parser<'a, u8, Vec<Token>> {
    let expr = number() | operator() | variable() | equal() | right_paren() | left_paren();
    expr.repeat(1..)
}

// pub fn equation_parser(input: String) -> Result<Vec<Token>, pom::Error> {
//     let string = input.to_owned();
//     let string_bytes = string.into_bytes();
//     expression()
//         .parse(&string_bytes)
// }

//fn parse_equation(equation: &str) -> Option<Vec<Token>> {
//let mut parsed_equation: Vec<Token> = Vec::new();

//for maybe_term in equation.split(' ') {
//let term = parse_term(maybe_term);
//parsed_equation.push(term);
//}

//let mut sides: Vec<Vec<Token>> = Vec::new();

//let parsed_equation_iter = parsed_equation.split(|el| el == &Token::Equal);

//for side in parsed_equation_iter {
//sides.push(side.to_vec());
//}

//let right: Vec<Token> = sides.pop().unwrap();

//let left: Vec<Token> = sides.pop().unwrap();

//dbg!(right);
//dbg!(left);

////let new_left: Vec<Term> = Vec::new();

////for left_el in left_grouped.iter() {
////if matches!(left_el, Token::Var) {
////new_left.push(left_el);
////} else {
////right_grouped.push(left_el);
////}
////}

////while let Some(el) = constants_and_operators_left_tuples.pop() {
////let negated_el = negate(el);
////constants_and_operators_right_tuples.push(negated_el);
////}

////while let Some(el) = vars_right_tuples.pop() {
////let negated_el = negate(el);
////vars_left_tuples.push(negated_el);
////}

////collapse_right(constants_and_operators_right_tuples)
//0.0
//}

//fn collapse_right(right: Vec<(Option<Token>, Token)>) -> f32 {
//right.iter().fold(
//0.0,
//|acc_val, maybe_operator_and_constant| match maybe_operator_and_constant {
//(None, Token::Constant(val)) => acc_val + *val,
//(Some(Token::Add), Token::Constant(val)) => acc_val + *val,
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
//(Some(Token::Add), Token::Constant(val)) => (Some(Token::Minus), Token::Constant(val)),
//(Some(Token::Minus), Token::Constant(val)) => (Some(Token::Add), Token::Constant(val)),
//(Some(Token::Multiplication), Token::Constant(val)) => {
//(Some(Token::Division), Token::Constant(val))
//}
//(Some(Token::Division), Token::Constant(val)) => {
//(Some(Token::Multiplication), Token::Constant(val))
//}
//(Some(Token::Add), Token::Var(val)) => (Some(Token::Minus), Token::Var(val)),
//(Some(Token::Minus), Token::Var(val)) => (Some(Token::Add), Token::Var(val)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equation_parser() {
        let input = b"2.0x+3.0=4.0^2.0";
        let expected = vec![
            Token::Constant(2.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(3.0),
            Token::Equal,
            Token::Constant(4.0),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(equation_parser(input), expected);
    }
}
