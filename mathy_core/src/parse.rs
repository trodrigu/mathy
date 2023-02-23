extern crate dimensioned as dim;
use dim::dimensions::Length;
use dim::si::{self, f32consts, Meter, FT, M, S, SI};
use dim::Dimensioned;
use nalgebra::dvector;

use core::panic;
use derivative::*;
use nalgebra::{DMatrix, DVector};
use num::{Signed, Zero};
use pom::parser::Parser;
use pom::parser::*;
use std::fmt::{Display, Formatter, Result as Res};

#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Token {
    Var(String),
    //Op(Operator),
    Add(Box<Self>, Box<Self>),
    Subtract(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Divide(Box<Self>, Box<Self>),
    Exponent(Box<Self>, Box<Self>),
    Modulo(Box<Self>, Box<Self>),
    //Constant(f32, unit),
    Complex(Complex, Option<Box<Self>>),
    Unit(Meter<f32>),
    //Equal,
    //RightParen,
    //LeftParen,
    //Function(String,
    //#[derivative(PartialEq = "ignore", Debug = "ignore")] Rc<Function>),
    Function(String, Vec<Self>, Box<Self>),
    FunctionValue(String, Vec<Self>),
    Vector(Vector),
    Matrix(Matrix),
}

pub type Vector = DVector<Complex>;
pub type Matrix = DMatrix<Complex>;

//#[derive(Derivative)]
//#[derivative(PartialEq, Eq, Clone, Debug)]
//pub enum Unit {
//Feet(Foot),
//Meters(Meter),
//}

//pub type Unit = SI<V, U>;

//pub type Unit =

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> Res {
        match self {
            Token::Complex(z, None) => {
                if z.im.is_zero() {
                    write!(f, "{}", z.re.clone())
                } else {
                    write!(
                        f,
                        "{} {} {}i",
                        z.re.clone(),
                        if z.im.is_negative() { "-" } else { "+" },
                        z.im.abs()
                    )
                }
            }
            Token::Vector(vec) => write!(f, "{}", vec),
            Token::Matrix(matrix) => write!(f, "{}", matrix),
            Token::Var(var_name) => write!(f, "{}", var_name),
            _ => todo!("woops"),
        }
    }
}

pub type Complex = num::complex::Complex<f32>;
//pub type

#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Type {
    Number(Complex),
    NumberWithUnit(Meter<f32>),
    Arithmetic,
    VectorComplex(DVector<Complex>),
    MatrixComplex(DMatrix<Complex>),
    Unknown,
}

fn matrix<'a>() -> Parser<'a, u8, Token> {
    let elems = list(number(), sym(b',') * space());
    let row = sym(b'[') * space() + elems - sym(b']');
    let matrix = sym(b'[') * space() + list(row, sym(b',') * space()) - sym(b']');
    matrix.name("matrix").map(|(_left_bracket, mat)| {
        dbg!(mat.clone());
        let matrix_of_complexes = mat
            .iter()
            .map(|(_, row)| {
                row.iter()
                    .map(|t| {
                        if let Token::Complex(c, _) = t {
                            c.clone()
                        } else {
                            panic!("not a token complex")
                        }
                    })
                    .collect::<Vec<Complex>>()
            })
            .collect::<Vec<Vec<Complex>>>();

        let row_count = matrix_of_complexes.len();
        let col_count = if let Some(first) = matrix_of_complexes.first() {
            first.len()
        } else {
            panic!("matrix has no columns");
        };
        dbg!(row_count.clone());
        dbg!(col_count.clone());

        let flat_matrix_of_complexes = matrix_of_complexes
            .into_iter()
            .flatten()
            .collect::<Vec<Complex>>();

        dbg!(flat_matrix_of_complexes.clone());
        Token::Matrix(DMatrix::from_vec(
            row_count,
            col_count,
            flat_matrix_of_complexes,
        ))
    })
}
fn vector<'a>() -> Parser<'a, u8, Token> {
    let elems = list(number(), sym(b',') * space());
    let p = sym(b'[') * space() + elems - sym(b']');
    p.name("vector").map(|(_left_bracket, vec)| {
        dbg!(vec.clone());
        let vec_of_complexes = vec
            .iter()
            .map(|token_complex| {
                if let Token::Complex(c, _) = token_complex {
                    c.clone()
                } else {
                    panic!("not a token complex")
                }
            })
            .collect::<Vec<Complex>>();

        Token::Vector(DVector::from(vec_of_complexes))
    })
}

fn variable<'a>() -> Parser<'a, u8, Token> {
    // TODO: remove ft or meters?
    one_of(b"abcdeghijklnopqrsuvwxyz")
        .repeat(1..)
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
        .name("variable")
        .map(Token::Var)
}

fn number_with_unit<'a>() -> Parser<'a, u8, Token> {
    let p = number() + (seq(b"ft") | seq(b"m")).opt();
    p.name("number_with_unit").map(|(num, unit)| match num {
        Token::Complex(num, _none) => match unit {
            Some(b"ft") => {
                let ft_in_m = num.re * f32consts::FT;
                Token::Complex(
                    Complex::new(ft_in_m.value_unsafe().clone(), 0.0),
                    Some(Box::new(Token::Unit(ft_in_m))),
                )
            }
            Some(b"m") => {
                let in_m = num.re * f32consts::M;
                Token::Complex(
                    Complex::new(in_m.value_unsafe().clone(), 0.0),
                    Some(Box::new(Token::Unit(in_m))),
                )
            }
            _ => Token::Complex(num, None),
        },
        _ => panic!("something went wrong!"),
    })
}

fn number<'a>() -> Parser<'a, u8, Token> {
    let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
    let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
    let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
    let number = integer + frac.opt() + exp.opt();
    number
        .name("number")
        .collect()
        .convert(|v| String::from_utf8(v.to_vec()))
        .convert(|s| s.parse::<Complex>())
        .map(|t| Token::Complex(t, None))
}

/// Token Related Parsers

fn trailing_atomic_expr<'a>() -> Parser<'a, u8, Token> {
    let p = number() - space() + one_of(b"+-*/^%") - space() + sym(b'(') - space()
        + call(expression)
        - space()
        + sym(b')');
    p.name("trailing_atomic_expr")
        .map(
            |((((expr1, op), _left_paren), expr2), _right_paren)| match op {
                b'^' => match expr2 {
                    Token::Multiply(l, r) => {
                        Token::Exponent(Box::new(expr1), Box::new(Token::Multiply(l, r)))
                    }
                    Token::Divide(l, r) => {
                        Token::Exponent(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    }
                    Token::Add(l, r) => {
                        Token::Exponent(Box::new(expr1), Box::new(Token::Add(l, r)))
                    }
                    Token::Subtract(l, r) => {
                        Token::Exponent(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    }
                    _ => todo!("nooo"),
                },
                b'*' => match expr2 {
                    Token::Divide(l, r) => {
                        Token::Multiply(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    }
                    Token::Add(l, r) => {
                        Token::Multiply(Box::new(expr1), Box::new(Token::Add(l, r)))
                    }
                    Token::Subtract(l, r) => {
                        Token::Multiply(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    }
                    _ => todo!("nooo"),
                },
                b'/' => match expr2 {
                    Token::Multiply(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Multiply(l, r)))
                    }
                    Token::Add(l, r) => Token::Divide(Box::new(expr1), Box::new(Token::Add(l, r))),
                    Token::Subtract(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    }
                    _ => todo!("nooo"),
                },
                b'+' => match expr2 {
                    Token::Multiply(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Multiply(l, r)))
                    }
                    Token::Divide(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    }
                    Token::Subtract(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    }
                    _ => todo!("nooo"),
                },
                _ => todo!("noooo"),
            },
        )
}
fn leading_atomic_expr<'a>() -> Parser<'a, u8, Token> {
    let p = sym(b'(') - space() + call(expression) - space() + sym(b')') - space()
        + one_of(b"+-*/^%")
        - space()
        + call(expression);
    p.name("leading_atomic_expr").map(
        |((((_left_paren, expr1), _right_paren), op), expr2)| match op {
            b'^' => match expr2 {
                Token::Complex(c, None) => {
                    Token::Exponent(Box::new(expr1), Box::new(Token::Complex(c, None)))
                }
                // TODO: refactor todos to a proper Error
                _ => todo!("nooo"),
            },
            b'/' => match expr2 {
                Token::Complex(c, None) => {
                    Token::Divide(Box::new(expr1), Box::new(Token::Complex(c, None)))
                }
                _ => todo!("nooo"),
            },
            b'*' => match expr2 {
                Token::Complex(c, None) => {
                    Token::Multiply(Box::new(expr1), Box::new(Token::Complex(c, None)))
                }
                _ => todo!("nooo"),
            },
            b'+' => match expr2 {
                Token::Complex(c, None) => {
                    Token::Add(Box::new(expr1), Box::new(Token::Complex(c, None)))
                }
                _ => todo!("nooo"),
            },
            _ => todo!("noooo"),
        },
    )
}

fn space<'a>() -> Parser<'a, u8, ()> {
    one_of(b" \t\r\n").repeat(0..).discard()
}

fn operator_vector<'a>() -> Parser<'a, u8, Token> {
    let parser = vector() - space() + one_of(b"+-*/^%") - space() + call(expression);
    parser
        .name("operator_vector")
        .map(|((left_vector, op), right_vector)| match op {
            b'+' => Token::Add(Box::new(left_vector), Box::new(right_vector)),
            _ => Token::Add(Box::new(left_vector), Box::new(right_vector)),
        })
}

fn operator_matrix<'a>() -> Parser<'a, u8, Token> {
    let parser = matrix() - space() + one_of(b"+-*/^%") - space() + call(expression);
    parser
        .name("operator_matrix")
        .map(|((left_matrix, op), right_matrix)| match op {
            b'+' => Token::Add(Box::new(left_matrix), Box::new(right_matrix)),
            _ => Token::Add(Box::new(left_matrix), Box::new(right_matrix)),
        })
}
fn operator<'a>() -> Parser<'a, u8, Token> {
    let parser = number_with_unit().opt() - space() + variable().opt() - space()
        + one_of(b"+-*/^%")
        - space()
        + call(expression);
    parser
        .name("regular_operator")
        .map(|(((left_maybe, v), op), r)| {
            let l: Token = if let Some(left_number) = left_maybe {
                if let Some(var) = v {
                    Token::Multiply(Box::new(left_number), Box::new(var))
                } else {
                    left_number
                }
            } else {
                if let Some(var) = v {
                    var
                } else {
                    panic!("no number or var!")
                }
            };

            match op {
                b'+' => Token::Add(Box::new(l), Box::new(r)),
                b'-' => Token::Subtract(Box::new(l), Box::new(r)),
                b'^' => match (l, r) {
                    // TODO: test left being other than complex
                    (Token::Complex(c1, None), Token::Complex(c2, None)) => Token::Exponent(
                        Box::new(Token::Complex(c1, None.clone())),
                        Box::new(Token::Complex(c2, None.clone())),
                    ),
                    (Token::Complex(c1, None), Token::Var(c2)) => Token::Exponent(
                        Box::new(Token::Complex(c1, None.clone())),
                        Box::new(Token::Var(c2.clone())),
                    ),
                    (Token::Complex(c1, None), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1, None.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1, None), Token::Subtract(inner_l, inner_r)) => {
                        Token::Subtract(
                            Box::new(Token::Exponent(
                                Box::new(Token::Complex(c1, None.clone())),
                                inner_l,
                            )),
                            inner_r,
                        )
                    }
                    (Token::Complex(c1, None), Token::Multiply(inner_l, inner_r)) => {
                        Token::Multiply(
                            Box::new(Token::Exponent(
                                Box::new(Token::Complex(c1, None.clone())),
                                inner_l,
                            )),
                            inner_r,
                        )
                    }
                    (Token::Complex(c1, None), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1, None.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),

                    // do it for the vars
                    (Token::Var(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Exponent(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Exponent(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Multiply(inner_l, inner_r)) => Token::Multiply(
                        Box::new(Token::Exponent(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Exponent(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (l, r) => {
                        todo!("whyyyy")
                    }
                },
                b'*' => match (l, r) {
                    (Token::Complex(c1, None), Token::Complex(c2, None)) => Token::Multiply(
                        Box::new(Token::Complex(c1.clone(), None)),
                        Box::new(Token::Complex(c2.clone(), None)),
                    ),
                    (Token::Complex(c1, None), Token::Var(c2)) => Token::Multiply(
                        Box::new(Token::Complex(c1.clone(), None)),
                        Box::new(Token::Var(c2.clone())),
                    ),
                    (Token::Complex(c1, None), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Multiply(
                            Box::new(Token::Complex(c1.clone(), None)),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1, None), Token::Subtract(inner_l, inner_r)) => {
                        Token::Subtract(
                            Box::new(Token::Multiply(
                                Box::new(Token::Complex(c1.clone(), None)),
                                inner_l,
                            )),
                            inner_r,
                        )
                    }
                    (Token::Complex(c1, None), Token::Multiply(inner_l, inner_r)) => {
                        Token::Multiply(
                            Box::new(Token::Multiply(
                                Box::new(Token::Complex(c1.clone(), None)),
                                inner_l,
                            )),
                            inner_r,
                        )
                    }

                    // do it for the vars
                    (Token::Var(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Multiply(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Multiply(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Multiply(inner_l, inner_r)) => Token::Multiply(
                        Box::new(Token::Multiply(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Multiply(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (l, r) => {
                        todo!("hermm")
                    }
                },
                b'/' => match (l, r) {
                    (Token::Complex(c1, None), Token::Complex(c2, None)) => Token::Divide(
                        Box::new(Token::Complex(c1.clone(), None)),
                        Box::new(Token::Complex(c2.clone(), None)),
                    ),
                    (Token::Complex(c1, None), Token::Var(c2)) => Token::Divide(
                        Box::new(Token::Complex(c1.clone(), None)),
                        Box::new(Token::Var(c2.clone())),
                    ),
                    (Token::Complex(c1, None), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Divide(
                            Box::new(Token::Complex(c1.clone(), None)),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1, None), Token::Subtract(inner_l, inner_r)) => {
                        Token::Subtract(
                            Box::new(Token::Divide(
                                Box::new(Token::Complex(c1.clone(), None)),
                                inner_l,
                            )),
                            inner_r,
                        )
                    }
                    (Token::Complex(c1, None), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Complex(c1.clone(), None)),
                        Box::new(Token::Divide(inner_l, inner_r)),
                    ),

                    // do it for the vars
                    (Token::Var(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Divide(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Divide(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    // divide is after multiply
                    (Token::Var(c1), Token::Multiply(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Multiply(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Var(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Divide(Box::new(Token::Var(c1.clone())), inner_l)),
                        inner_r,
                    ),

                    (l, r) => {
                        todo!("hi")
                    }
                },
                b'%' => Token::Modulo(Box::new(l), Box::new(r)),
                _ => Token::Exponent(Box::new(l), Box::new(r)),
            }
        })
}

fn function_value<'a>() -> Parser<'a, u8, Token> {
    // f(9)
    let f = function_name() - sym(b'(') + list(number().opt(), sym(b',') * space()) - sym(b')');
    f.name("function_value").map(|(f_name, found_args)| {
        let mut args = Vec::new();

        for v in found_args {
            if let Some(inner_v) = v {
                args.push(inner_v);
            }
        }
        Token::FunctionValue(f_name, args)
    })
}

// this is like value in json parser
fn expression<'a>() -> Parser<'a, u8, Token> {
    trailing_atomic_expr()
        | leading_atomic_expr()
        | operator()
        | function_value()
        | number_with_unit()
        | variable()
        | operator_vector()
        | operator_matrix()
        | matrix()
        | vector()
}

pub fn total_expr<'a>() -> Parser<'a, u8, Token> {
    expression() - end()
}

/// Action Related Parsers

#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Action {
    DefineFunc(String, Vec<Token>, Token),
    DefineVar(String, Token),
    EvalExpr(Token),
}

fn function_name<'a>() -> Parser<'a, u8, String> {
    let function_name = one_of(b"abcdefghijklmnopqrstuvwxyz").repeat(1..)
        | one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ").repeat(1..);
    function_name
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
}

fn define_var<'a>() -> Parser<'a, u8, Action> {
    let f = variable() - space() - sym(b'=') - space() + call(expression);
    f.map(|(var_name, f)| Action::DefineVar(var_name.to_string(), f))
}

fn define_func<'a>() -> Parser<'a, u8, Action> {
    let f = function_name() - sym(b'(') + list(variable().opt(), sym(b',') * space())
        - sym(b')')
        - sym(b'=')
        + call(expression);
    f.map(|((fname, found_vars), f)| {
        let mut vars = Vec::new();

        for v in found_vars {
            if let Some(inner_v) = v {
                vars.push(inner_v);
            }
        }

        Action::DefineFunc(fname, vars, f)
    })
}

fn eval_expr<'a>() -> Parser<'a, u8, Action> {
    expression().map(|expr| Action::EvalExpr(expr))
}

pub fn total_action<'a>() -> Parser<'a, u8, Action> {
    (define_var() | define_func() | eval_expr()) - end()
}

#[cfg(test)]
mod tests {
    use crate::parse::{total_action, Action, Complex, Token};

    #[test]
    fn test_define_func() {
        p(
            b"f(x)=2x+1",
            Action::DefineFunc(
                "f".to_string(),
                vec![Token::Var("x".to_string())],
                Token::Add(
                    Box::new(Token::Multiply(
                        Box::new(real_num(2.0)),
                        Box::new(Token::Var("x".to_string())),
                    )),
                    Box::new(real_num(1.0)),
                ),
            ),
        );

        p(
            b"f(x)=x^2+1",
            Action::DefineFunc(
                "f".to_string(),
                vec![Token::Var("x".to_string())],
                Token::Add(
                    Box::new(Token::Exponent(
                        Box::new(Token::Var("x".to_string())),
                        Box::new(real_num(2.0)),
                    )),
                    Box::new(real_num(1.0)),
                ),
            ),
        );

        p(
            b"f(x)=x*3+3",
            Action::DefineFunc(
                "f".to_string(),
                vec![Token::Var("x".to_string())],
                Token::Add(
                    Box::new(Token::Multiply(
                        Box::new(Token::Var("x".to_string())),
                        Box::new(real_num(3.0)),
                    )),
                    Box::new(real_num(3.0)),
                ),
            ),
        );

        p(
            b"f(x)=2/x",
            Action::DefineFunc(
                "f".to_string(),
                vec![Token::Var("x".to_string())],
                Token::Divide(
                    Box::new(real_num(2.0)),
                    Box::new(Token::Var("x".to_string())),
                ),
            ),
        );
    }

    #[test]
    fn test_define_var() {
        p(
            b"x = 2.0",
            Action::DefineVar("x".to_string(), real_num(2.0)),
        );
    }

    #[test]
    fn test_eval_expr() {
        p(
            b"2.0*2.0+1.0",
            Action::EvalExpr(Token::Add(
                Box::new(Token::Multiply(
                    Box::new(real_num(2.0)),
                    Box::new(real_num(2.0)),
                )),
                Box::new(real_num(1.0)),
            )),
        );
    }

    #[track_caller]
    fn p(string: &'static [u8], action: Action) {
        assert_eq!(total_action().parse(string), Ok(action));
    }

    fn real_num(n: f32) -> Token {
        Token::Complex(Complex::new(n, 0.0), None)
    }
}
