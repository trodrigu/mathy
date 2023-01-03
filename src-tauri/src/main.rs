#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use pom::parser::Parser;
use pom::parser::*;
use std::rc::Rc;

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
    Function(String, Option<Rc<Token>>),
    FunctionValue(String, f32),
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

fn eval(tokens: Vec<Token>) -> f32 {
    let parsed_equation_iter = tokens.split(|el| el == &Token::Equal);

    let mut sides: Vec<Vec<Token>> = Vec::new();
    for side in parsed_equation_iter {
        sides.push(side.to_vec());
    }

    let right: Vec<Token> = sides.pop().unwrap();

    collapse_right(right)
}

fn function<'a>() -> Parser<'a, u8, Token> {
    let f = function_name() - left_paren() + variable().opt() - right_paren();

    f.map(|(fname, v)| match v {
        Some(inner_v) => Token::Function(fname, Some(Rc::new(inner_v))),
        None => Token::Function(fname, None),
    })
}
fn function_name<'a>() -> Parser<'a, u8, String> {
    let function_name = one_of(b"abcdefghijklmnopqrstuvwxyz").repeat(1..)
        | one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ").repeat(1..);
    function_name
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
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
    let number = integer + frac.opt() + exp.opt();
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
    let expr =
        function() | number() | operator() | variable() | equal() | right_paren() | left_paren();
    expr.repeat(1..)
}

fn substitute(equation: Vec<Token>, value: f32) -> Vec<Token> {
    // let equation_used_for_find = equation.clone();
    let function = equation
        .iter()
        .find(|&el| matches!(el, Token::Function(_, _)))
        .unwrap();

    let var = match function {
        Token::Function(_function_name, Some(rc_var)) => rc_var.as_ref(),
        _ => todo!(),
    };

    let var_name = match var {
        Token::Var(name) => name,
        _ => todo!(),
    };

    let mut new_equation: Vec<Token> = Vec::new();
    let mut eq_iter = equation.iter();
    while let Some(token) = eq_iter.next() {
        match token {
            Token::Var(name) => {
                if name == var_name {
                    new_equation.push(Token::LeftParen);
                    new_equation.push(Token::Constant(value));
                    new_equation.push(Token::RightParen);
                }
            }
            _ => new_equation.push(token.clone()),
        }
    }

    new_equation
        .iter()
        .map(|el| match el {
            Token::Function(name, _var) => Token::FunctionValue(name.clone(), value),
            _ => el.clone(),
        })
        .collect::<Vec<Token>>()
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
//
//
fn eval_divide(equation: Vec<Token>) -> Vec<Token> {
    eval_binary_operation(equation, Token::Op(Operator::Divide))
}

fn eval_multiply(equation: Vec<Token>) -> Vec<Token> {
    let mut equation_iter = equation.iter();
    let mut new_equation = Vec::new();
    while let Some(token) = equation_iter.next() {
        match token {
            Token::LeftParen => match equation_iter.next() {
                Some(Token::Constant(left)) => {
                    let possible_paren_or_constant = equation_iter.next();
                    dbg!(possible_paren_or_constant.clone());

                    match possible_paren_or_constant {
                        Some(Token::RightParen) => match equation_iter.next() {
                            Some(Token::LeftParen) => {
                                if let Some(Token::Constant(right)) = equation_iter.next() {
                                    let constant_value = left * right;
                                    let _unneeded_right_paren = equation_iter.next();
                                    new_equation.push(Token::Constant(constant_value));
                                };
                            }
                            Some(Token::Constant(right)) => {
                                let constant_value = left * right;
                                new_equation.push(Token::Constant(constant_value));
                            }
                            Some(_) => todo!(),
                            None => todo!(),
                        },
                        Some(Token::Constant(left)) => {
                            if let Some(Token::Constant(right)) = equation_iter.next() {
                                let constant_value = left * right;
                                new_equation.push(Token::Constant(constant_value));
                            };
                        }
                        Some(_) => todo!(),
                        None => {}
                    }
                }
                None => todo!(),
                Some(_) => todo!(),
            },
            Token::Constant(left) => match equation_iter.next() {
                Some(Token::LeftParen) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left * right;
                        let _unneeded_right_paren = equation_iter.next();
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(Token::Constant(left)) => {
                    if let Some(Token::Constant(right)) = equation_iter.next() {
                        let constant_value = left * right;
                        new_equation.push(Token::Constant(constant_value));
                    };
                }
                Some(Token::Op(Operator::Multiply)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left * *right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(_) => todo!(),
                None => todo!(),
            },
            _ => new_equation.push(token.clone()),
        }
    }
    new_equation
}

fn eval_add(equation: Vec<Token>) -> Vec<Token> {
    eval_binary_operation(equation, Token::Op(Operator::Add))
}

fn eval_binary_operation(equation: Vec<Token>, operator: Token) -> Vec<Token> {
    let mut equation_iter = equation.iter();
    let mut new_equation = Vec::new();
    while let Some(token) = equation_iter.next() {
        match token {
            Token::Constant(left) => match equation_iter.next() {
                Some(Token::Op(Operator::Divide)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left / right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(Token::Op(Operator::Add)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left + right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(_) => todo!(),
                None => todo!(),
            },
            Token::Op(Operator::Divide) => match equation_iter.next() {
                Some(Token::Constant(left)) => match new_equation.pop() {
                    Some(Token::Constant(last_val)) => {
                        let constant_value = last_val / left;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(_) => todo!(),
                None => todo!(),
            },
            Token::Op(Operator::Add) => match equation_iter.next() {
                Some(Token::Constant(left)) => match new_equation.pop() {
                    Some(Token::Constant(last_val)) => {
                        let constant_value = last_val + left;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(_) => todo!(),
                None => todo!(),
            },
            _ => new_equation.push(token.clone()),
        }
    }
    new_equation
}

fn eval_exponent(equation: Vec<Token>) -> Vec<Token> {
    let mut equation_iter = equation.iter();
    let mut new_equation = Vec::new();
    while let Some(token) = equation_iter.next() {
        match token {
            Token::Op(Operator::Exponent) => {
                if let Some(Token::Constant(power)) = equation_iter.next() {
                    let previous_token = new_equation.pop();

                    match previous_token {
                        Some(Token::RightParen) => {
                            if let Some(Token::Constant(base)) = new_equation.pop() {
                                let constant_value = base.powf(*power);
                                new_equation.push(Token::Constant(constant_value));
                                new_equation.push(Token::RightParen);
                            }
                        }
                        Some(Token::Constant(base)) => {
                            let constant_value = base.powf(*power);
                            new_equation.push(Token::Constant(constant_value));
                        }
                        Some(_) => todo!(),
                        None => todo!(),
                    }
                }
            }
            _ => new_equation.push(token.clone()),
        }
    }
    new_equation
    //  right.iter().fold(
    //      (None, 0.0),
    //      |acc_val, token| match token {
    //          Token::Constant(val) => {
    //             match acc_val {
    //                (Some(Token::Op(Operator::Add)),running_total) => (None, running_total + val),
    //                (Some(Token::Op(Operator::Subtract)),running_total) => (None, running_total - val),
    //                (Some(Token::Op(Operator::Multiply)),running_total) => (None, running_total * val),
    //                (Some(Token::Op(Operator::Divide)),running_total) => (None, running_total / val),

    //                (possible_op,running_total) => (possible_op, running_total),
    //             }
    //          },
    //          Token::Op(op) => {
    //             match acc_val {
    //                (None,running_total) => (Some(Token::Op(op.clone())), running_total),
    //                (Some(_), _) => todo!(),
    //             }
    //          },
    //          _ => acc_val,
    //      },
    // //  ).1
}
fn collapse_right(right: Vec<Token>) -> f32 {
    // how do we apply parenthesis, exponent, multiply, division, addition, subtract
    right
        .iter()
        .fold((None, 0.0), |acc_val, token| match token {
            Token::Constant(val) => match acc_val {
                (Some(Token::Op(Operator::Add)), running_total) => (None, running_total + val),
                (Some(Token::Op(Operator::Subtract)), running_total) => (None, running_total - val),
                (Some(Token::Op(Operator::Multiply)), running_total) => (None, running_total * val),
                (Some(Token::Op(Operator::Divide)), running_total) => (None, running_total / val),

                (possible_op, running_total) => (possible_op, running_total),
            },
            Token::Op(op) => match acc_val {
                (None, running_total) => (Some(Token::Op(op.clone())), running_total),
                (Some(_), _) => todo!(),
            },
            _ => acc_val,
        })
        .1
}

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
    fn test_parse_define_polynomial_with_base_case() {
        let input = b"f(x)=2.0x+3.0-4.0^2.0";
        let expected = vec![
            Token::Function("f".to_string(), Some(Rc::new(Token::Var("x".to_string())))),
            Token::Equal,
            Token::Constant(2.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(3.0),
            Token::Op(Operator::Subtract),
            Token::Constant(4.0),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(expression().parse(input), Ok(expected));
    }

    #[test]
    fn test_parse_define_polynomial_with_double_neg() {
        let input = b"f(x)=2.0x+3.0--4.0^2.0";
        let expected = vec![
            Token::Function("f".to_string(), Some(Rc::new(Token::Var("x".to_string())))),
            Token::Equal,
            Token::Constant(2.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(3.0),
            Token::Op(Operator::Subtract),
            Token::Op(Operator::Subtract),
            Token::Constant(4.0),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(expression().parse(input), Ok(expected));
    }

    #[test]
    fn test_substitute_polynomial_with_double_neg() {
        let input = vec![
            Token::Function("f".to_string(), Some(Rc::new(Token::Var("x".to_string())))),
            Token::Equal,
            Token::Constant(2.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(3.0),
            Token::Op(Operator::Subtract),
            Token::Constant(4.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(
            substitute(input, 9.0),
            vec![
                Token::FunctionValue("f".to_string(), 9.0),
                Token::Equal,
                Token::Constant(2.0),
                Token::LeftParen,
                Token::Constant(9.0),
                Token::RightParen,
                Token::Op(Operator::Add),
                Token::Constant(3.0),
                Token::Op(Operator::Subtract),
                Token::Constant(4.0),
                Token::LeftParen,
                Token::Constant(9.0),
                Token::RightParen,
                Token::Op(Operator::Exponent),
                Token::Constant(2.0),
            ]
        );
    }

    #[test]
    fn test_eval_exponent_polynomial() {
        let input = vec![
            Token::Constant(9.0),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_exponent(input), vec![Token::Constant(81.0),]);
    }

    #[test]
    fn test_eval_exponent_polynomial_with_variable_sub() {
        let input = vec![
            Token::LeftParen,
            Token::Constant(9.0),
            Token::RightParen,
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(
            eval_exponent(input),
            vec![Token::LeftParen, Token::Constant(81.0), Token::RightParen,]
        );
    }

    #[test]
    fn test_eval_polynomial_double_neg() {
        let input = vec![
            Token::Function("f".to_string(), Some(Rc::new(Token::Var("x".to_string())))),
            Token::Equal,
            Token::Constant(2.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(3.0),
            Token::Op(Operator::Subtract),
            Token::Constant(4.0),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
        ];
        assert_eq!(eval(input), -9.0);
    }

    #[test]
    fn test_eval_multiply() {
        let input = vec![
            Token::Constant(9.0),
            Token::Op(Operator::Multiply),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_multiply(input), vec![Token::Constant(18.0),]);
    }

    #[test]
    fn test_eval_multiply_parens() {
        let input = vec![
            Token::LeftParen,
            Token::Constant(9.0),
            Token::RightParen,
            Token::Constant(2.0),
        ];
        assert_eq!(eval_multiply(input), vec![Token::Constant(18.0),]);

        let input = vec![
            Token::Constant(9.0),
            Token::LeftParen,
            Token::Constant(2.0),
            Token::RightParen,
        ];
        assert_eq!(eval_multiply(input), vec![Token::Constant(18.0),]);

        let input = vec![
            Token::LeftParen,
            Token::Constant(9.0),
            Token::RightParen,
            Token::LeftParen,
            Token::Constant(2.0),
            Token::RightParen,
        ];
        assert_eq!(eval_multiply(input), vec![Token::Constant(18.0),]);
    }

    #[test]
    fn test_eval_division() {
        let input = vec![
            Token::Constant(9.0),
            Token::Op(Operator::Divide),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_divide(input), vec![Token::Constant(4.5),]);

        let input = vec![
            Token::Constant(100.0),
            Token::Op(Operator::Divide),
            Token::Constant(2.0),
            Token::Op(Operator::Divide),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_divide(input), vec![Token::Constant(25.0),]);
    }

    #[test]
    fn test_eval_addition() {
        let input = vec![
            Token::Constant(9.0),
            Token::Op(Operator::Add),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_add(input), vec![Token::Constant(11.0),]);

        let input = vec![
            Token::Constant(100.0),
            Token::Op(Operator::Add),
            Token::Constant(2.0),
            Token::Op(Operator::Add),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_add(input), vec![Token::Constant(104.0),]);
    }
}
