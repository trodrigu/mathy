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
    match result {
        Ok(tokens) => eval(tokens, 2.0).to_string(),
        Err(_err) => {
            panic!("nope")
        }
    }
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

fn eval(tokens: Vec<Token>, value: f32) -> f32 {
    collapse_right(tokens, value)
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

fn eval_multiply(equation: Vec<Token>) -> Vec<Token> {
    let mut equation_iter = equation.iter();
    let mut new_equation = Vec::new();
    while let Some(token) = equation_iter.next() {
        match token {
            Token::LeftParen => match equation_iter.next() {
                Some(Token::Constant(left)) => match equation_iter.next() {
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
                        Some(t) => {
                            new_equation.push(Token::LeftParen);
                            new_equation.push(Token::Constant(left.clone()));
                            new_equation.push(Token::RightParen);
                            new_equation.push(t.clone());
                        }
                        None => {
                            new_equation.push(Token::LeftParen);
                            new_equation.push(Token::Constant(left.clone()));
                            new_equation.push(Token::RightParen);
                        }
                    },
                    Some(Token::Constant(left)) => match equation_iter.next() {
                        Some(Token::Constant(right)) => {
                            let constant_value = left * right;
                            new_equation.push(Token::Constant(constant_value));
                        }
                        Some(_) => todo!(),
                        None => todo!(),
                    },
                    Some(_) => todo!(),
                    None => new_equation.push(Token::Constant(left.clone())),
                },
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
                Some(Token::Constant(left)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left * right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(Token::Op(Operator::Multiply)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left * *right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(t) => {
                    new_equation.push(Token::Constant(left.clone()));
                    new_equation.push(t.clone())
                }
                None => new_equation.push(Token::Constant(left.clone())),
            },
            t => new_equation.push(t.clone()),
        }
    }
    new_equation
}

fn eval_divide(equation: Vec<Token>) -> Vec<Token> {
    eval_binary_operation(equation)
}

fn eval_add(equation: Vec<Token>) -> Vec<Token> {
    eval_binary_operation(equation)
}

fn eval_subtract(equation: Vec<Token>) -> Vec<Token> {
    eval_binary_operation(equation)
}

fn eval_binary_operation(equation: Vec<Token>) -> Vec<Token> {
    let mut equation_iter = equation.iter();
    let mut new_equation = Vec::new();
    while let Some(token) = equation_iter.next() {
        match token {
            Token::LeftParen => match equation_iter.next() {
                Some(Token::Constant(left)) => match equation_iter.next() {
                    Some(Token::RightParen) => match equation_iter.next() {
                        Some(Token::Op(Operator::Divide)) => match equation_iter.next() {
                            Some(Token::LeftParen) => {
                                if let Some(Token::Constant(right)) = equation_iter.next() {
                                    let constant_value = left / right;
                                    let _unneeded_right_paren = equation_iter.next();
                                    new_equation.push(Token::Constant(constant_value));
                                };
                            }
                            Some(Token::Constant(right)) => {
                                let constant_value = left / right;
                                new_equation.push(Token::Constant(constant_value));
                            }
                            Some(_) => todo!(),
                            None => todo!(),
                        },
                        Some(Token::Op(Operator::Add)) => match equation_iter.next() {
                            Some(Token::LeftParen) => {
                                if let Some(Token::Constant(right)) = equation_iter.next() {
                                    let constant_value = left + right;
                                    let _unneeded_right_paren = equation_iter.next();
                                    new_equation.push(Token::Constant(constant_value));
                                };
                            }
                            Some(Token::Constant(right)) => {
                                let constant_value = left + right;
                                new_equation.push(Token::Constant(constant_value));
                            }
                            Some(t) => new_equation.push(t.clone()),
                            None => todo!(),
                        },
                        Some(Token::Op(Operator::Subtract)) => match equation_iter.next() {
                            Some(Token::LeftParen) => {
                                if let Some(Token::Constant(right)) = equation_iter.next() {
                                    let constant_value = left - right;
                                    let _unneeded_right_paren = equation_iter.next();
                                    new_equation.push(Token::Constant(constant_value));
                                };
                            }
                            Some(Token::Constant(right)) => {
                                let constant_value = left - right;
                                new_equation.push(Token::Constant(constant_value));
                            }
                            Some(_) => todo!(),
                            None => todo!(),
                        },
                        Some(t) => {
                            new_equation.push(Token::LeftParen);
                            new_equation.push(Token::Constant(left.clone()));
                            new_equation.push(Token::RightParen);
                            new_equation.push(t.clone());
                        }
                        None => {
                            new_equation.push(Token::LeftParen);
                            new_equation.push(Token::Constant(left.clone()));
                            new_equation.push(Token::RightParen);
                        }
                    },
                    Some(_) => todo!(),
                    None => panic!("nope"),
                },
                Some(_) => todo!(),
                None => panic!("nope"),
            },
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
                    Some(t) => new_equation.push(t.clone()),
                    None => todo!(),
                },
                Some(Token::Op(Operator::Subtract)) => match equation_iter.next() {
                    Some(Token::Constant(right)) => {
                        let constant_value = left - right;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(t) => {
                    todo!();
                }
                None => new_equation.push(Token::Constant(left.clone())),
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
                Some(Token::LeftParen) => match equation_iter.next() {
                    Some(Token::Constant(left)) => match new_equation.pop() {
                        Some(Token::Constant(last_val)) => {
                            let constant_value = last_val / left;
                            new_equation.push(Token::Constant(constant_value));
                            let _unneeded_right_paren = equation_iter.next();
                        }
                        Some(_) => todo!(),
                        None => todo!(),
                    },
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
                Some(Token::LeftParen) => match equation_iter.next() {
                    Some(Token::Constant(left)) => match new_equation.pop() {
                        Some(Token::Constant(last_val)) => {
                            let constant_value = last_val + left;
                            new_equation.push(Token::Constant(constant_value));
                            let _unneeded_right_paren = equation_iter.next();
                        }
                        Some(_) => todo!(),
                        None => todo!(),
                    },
                    Some(_) => todo!(),
                    None => todo!(),
                },
                Some(t) => todo!(),
                None => todo!(),
            },
            Token::Op(Operator::Subtract) => match equation_iter.next() {
                Some(Token::Constant(left)) => match new_equation.pop() {
                    Some(Token::Constant(last_val)) => {
                        let constant_value = last_val - left;
                        new_equation.push(Token::Constant(constant_value));
                    }
                    Some(t) => new_equation.push(t),
                    None => todo!(),
                },
                Some(Token::LeftParen) => match equation_iter.next() {
                    Some(Token::Constant(left)) => match new_equation.pop() {
                        Some(Token::Constant(last_val)) => {
                            let constant_value = last_val + left;
                            new_equation.push(Token::Constant(constant_value));
                            let _unneeded_right_paren = equation_iter.next();
                        }
                        Some(_) => todo!(),
                        None => todo!(),
                    },
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
            Token::Op(Operator::Exponent) => match equation_iter.next() {
                Some(Token::Constant(power)) => {
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
                Some(_) => todo!(),
                None => todo!(),
            },
            _ => new_equation.push(token.clone()),
        }
    }
    new_equation
}
fn collapse_right(right: Vec<Token>, value: f32) -> f32 {
    let substituted = substitute(right, value);

    let mut sides: Vec<Vec<Token>> = Vec::new();

    let parsed_equation_iter = substituted.split(|el| el == &Token::Equal);

    for side in parsed_equation_iter {
        sides.push(side.to_vec());
    }

    let right: Vec<Token> = sides.pop().unwrap();

    let mut evaluated_tokens = evaluate_tokens(right, value);

    match evaluated_tokens.pop() {
        Some(Token::Constant(val)) => val,
        Some(Token::RightParen) => {
            if let Some(Token::Constant(val)) = evaluated_tokens.pop() {
                val
            } else {
                panic!("nope")
            }
        }
        Some(_) => todo!(),
        None => todo!(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum IndexedParen {
    LeftParen(usize),
    RightParen(usize),
}

fn evaluate_tokens(tokens: Vec<Token>, value: f32) -> Vec<Token> {
    let mut ts = tokens.clone();
    loop {
        let unmatched_pairs = ts.iter().enumerate().filter_map(|(i, x)| match x {
            Token::LeftParen => Some(IndexedParen::LeftParen(i)),
            Token::RightParen => Some(IndexedParen::RightParen(i)),
            _ => None,
        });

        let (_d, mut pairs) = unmatched_pairs.fold((0, Vec::new()), |acc, x| {
            let (mut current_depth, mut current_pairs) = acc;
            match x {
                IndexedParen::LeftParen(left_i) => {
                    if current_pairs.is_empty() {
                        current_pairs.push((Some(IndexedParen::LeftParen(left_i)), 0, None));
                        (current_depth, current_pairs)
                    } else {
                        current_depth += 1;
                        current_pairs.push((
                            Some(IndexedParen::LeftParen(left_i)),
                            current_depth,
                            None,
                        ));
                        (current_depth, current_pairs)
                    }
                }
                IndexedParen::RightParen(right_i) => {
                    if current_pairs.is_empty() {
                        panic!("No opening Left Paren!");
                    } else {
                        let pos = current_pairs
                            .iter()
                            .position(|(l_paren, d, r_paren)| {
                                *d == current_depth && r_paren.is_none()
                            })
                            .unwrap();
                        let mut_el = current_pairs.get_mut(pos).unwrap();
                        mut_el.2 = Some(IndexedParen::RightParen(right_i));

                        current_depth -= 1;
                        (current_depth, current_pairs)
                    }
                }
            }
        });

        let mut pairs_without_singles = pairs
            .into_iter()
            .filter(|x| {
                let (left_p, _d, right_p) = x;
                let left_i: usize = if let Some(IndexedParen::LeftParen(left_i)) = left_p {
                    *left_i
                } else {
                    panic!("nope");
                };
                let right_i: usize = if let Some(IndexedParen::RightParen(right_i)) = right_p {
                    *right_i
                } else {
                    panic!("nope")
                };

                let distance = right_i - left_i;
                distance > 2
            })
            .collect::<Vec<(Option<IndexedParen>, i32, Option<IndexedParen>)>>();

        if let Some((left_p, _d, right_p)) = pairs_without_singles.first_mut() {
            let left_i: usize = if let Some(IndexedParen::LeftParen(left_i)) = left_p {
                *left_i
            } else {
                panic!("nope");
            };
            let right_i: usize = if let Some(IndexedParen::RightParen(right_i)) = right_p {
                *right_i
            } else {
                panic!("nope")
            };
            let slice_to_eval = &ts[left_i + 1..right_i];
            let vec_to_eval = slice_to_eval.to_vec();
            let res = do_eval(vec_to_eval, value);

            ts.splice(left_i + 1..right_i, res);
        } else {
            break;
        }
    }

    do_eval(ts, value)
}

fn do_eval(tokens: Vec<Token>, value: f32) -> Vec<Token> {
    let exponents_evaluated = eval_exponent(tokens);
    let multiply_evaluated = eval_multiply(exponents_evaluated);
    let division_evaluated = eval_divide(multiply_evaluated);
    let addition_evaluated = eval_add(division_evaluated);
    eval_subtract(addition_evaluated)
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

        let input = vec![
            Token::LeftParen,
            Token::Constant(2.0),
            Token::RightParen,
            Token::Op(Operator::Exponent),
            Token::Constant(3.0),
        ];

        assert_eq!(
            eval_exponent(input),
            vec![Token::LeftParen, Token::Constant(8.0), Token::RightParen,]
        );
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

        // no multiply operator still runs fine
        let input = vec![
            Token::LeftParen,
            Token::Constant(9.0),
            Token::RightParen,
            Token::Op(Operator::Divide),
            Token::LeftParen,
            Token::Constant(2.0),
            Token::RightParen,
        ];
        assert_eq!(
            eval_multiply(input),
            vec![
                Token::LeftParen,
                Token::Constant(9.0),
                Token::RightParen,
                Token::Op(Operator::Divide),
                Token::LeftParen,
                Token::Constant(2.0),
                Token::RightParen,
            ]
        );
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

        let input = vec![
            Token::LeftParen,
            Token::Constant(100.0),
            Token::RightParen,
            Token::Op(Operator::Divide),
            Token::LeftParen,
            Token::Constant(2.0),
            Token::RightParen,
        ];
        assert_eq!(eval_divide(input), vec![Token::Constant(50.0)]);
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

        let input = vec![
            Token::LeftParen,
            Token::Constant(8.0),
            Token::RightParen,
            Token::Op(Operator::Add),
            Token::Constant(1.0),
            Token::Op(Operator::Add),
            Token::LeftParen,
            Token::Constant(1.0),
            Token::RightParen,
        ];
        assert_eq!(eval_add(input), vec![Token::Constant(10.0),]);
    }

    #[test]
    fn test_eval_subtract() {
        let input = vec![
            Token::Constant(9.0),
            Token::Op(Operator::Subtract),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_subtract(input), vec![Token::Constant(7.0),]);

        let input = vec![
            Token::Constant(100.0),
            Token::Op(Operator::Subtract),
            Token::Constant(2.0),
            Token::Op(Operator::Subtract),
            Token::Constant(2.0),
        ];
        assert_eq!(eval_subtract(input), vec![Token::Constant(96.0),]);
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
        assert_eq!(eval(input, 2.0), -9.0);
    }

    #[test]
    fn test_eval_fraction_parens_polynomial() {
        // f(x)=(4x^2-1)/(3x+4)
        let input = vec![
            Token::Function("f".to_string(), Some(Rc::new(Token::Var("x".to_string())))),
            Token::Equal,
            Token::LeftParen,
            Token::Constant(4.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Exponent),
            Token::Constant(2.0),
            Token::Op(Operator::Subtract),
            Token::Constant(1.0),
            Token::RightParen,
            Token::Op(Operator::Divide),
            Token::LeftParen,
            Token::Constant(3.0),
            Token::Var("x".to_string()),
            Token::Op(Operator::Add),
            Token::Constant(4.0),
            Token::RightParen,
        ];
        assert_eq!(eval(input, 2.0), 1.5);
    }
}
