use derivative::*;
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
    //Constant(f32),
    Complex(Complex),
    //Equal,
    //RightParen,
    //LeftParen,
    //Function(String,
    //#[derivative(PartialEq = "ignore", Debug = "ignore")] Rc<Function>),
    //Function(String, Rc<Token>),
    //FunctionValue(String, f32),
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> Res {
        match self {
            Token::Complex(z) => {
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
            _ => todo!("hi"),
        }
    }
}

pub type Complex = num::complex::Complex<f32>;

#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Type {
    Number(Complex),
    Arithmetic,
    Unknown,
}

fn variable<'a>() -> Parser<'a, u8, Token> {
    one_of(b"abcdefghijklmnopqrstuvwxyz")
        .repeat(1..)
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
        .name("variable")
        .map(Token::Var)
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
        .map(Token::Complex)
}

//
// more of an identifier currently!
// this is like object in json parser
//fn function<'a>() -> Parser<'a, u8, Token> {
//// we need to capture all tokens between the parens and use call to reinvoke expression
//let f = function_name() - left_paren() + variable().opt() - right_paren() - sym(b'=')
//+ call(expression);

//f.map(|((fname, var), f)| match var {
//Some(_var) => Token::Function(fname, Rc::new(f)),
//None => Token::Function(fname, Rc::new(f)),
//})
//}

fn function_name<'a>() -> Parser<'a, u8, String> {
    let function_name = one_of(b"abcdefghijklmnopqrstuvwxyz").repeat(1..)
        | one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ").repeat(1..);
    function_name
        .collect()
        .convert(|sy| (String::from_utf8(sy.to_vec())))
}

fn trailing_atomic_expr<'a>() -> Parser<'a, u8, Token> {
    let p = number() + one_of(b"+-*/^%") + sym(b'(') + call(expression) + sym(b')');
    p.name("trailing_atomic_expr")
        .map(
            |((((expr1, op), _left_paren), expr2), _right_paren)| match op {
                b'*' => match expr2 {
                    Token::Divide(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    }
                    Token::Add(l, r) => Token::Divide(Box::new(expr1), Box::new(Token::Add(l, r))),
                    Token::Subtract(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Subtract(l, r)))
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
    let p = sym(b'(') + call(expression) + sym(b')') + one_of(b"+-*/^%") + call(expression);
    p.name("leading_atomic_expr").map(
        |((((_left_paren, expr1), _right_paren), op), expr2)| match op {
            b'/' => match expr2 {
                Token::Complex(c) => Token::Divide(Box::new(expr1), Box::new(Token::Complex(c))),
                _ => todo!("nooo"),
            },
            b'*' => match expr2 {
                Token::Complex(c) => Token::Multiply(Box::new(expr1), Box::new(Token::Complex(c))),
                _ => todo!("nooo"),
            },
            b'+' => match expr2 {
                Token::Complex(c) => Token::Add(Box::new(expr1), Box::new(Token::Complex(c))),
                _ => todo!("nooo"),
            },
            _ => todo!("noooo"),
        },
    )
}

fn operator<'a>() -> Parser<'a, u8, Token> {
    let parser = number().opt() + variable().opt() + one_of(b"+-*/^%") + call(expression);
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
                b'*' => match (l, r) {
                    (Token::Complex(c1), Token::Complex(c2)) => Token::Multiply(
                        Box::new(Token::Complex(c1.clone())),
                        Box::new(Token::Complex(c2.clone())),
                    ),
                    (Token::Complex(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Multiply(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Multiply(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Multiply(inner_l, inner_r)) => Token::Multiply(
                        Box::new(Token::Multiply(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (l, r) => todo!("hi"),
                },
                b'/' => match (l, r) {
                    (Token::Complex(c1), Token::Complex(c2)) => Token::Divide(
                        Box::new(Token::Complex(c1.clone())),
                        Box::new(Token::Complex(c2.clone())),
                    ),
                    (Token::Complex(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)),
                        inner_r,
                    ),
                    (l, r) => todo!("hi"),
                },
                b'^' => Token::Exponent(Box::new(l), Box::new(r)),
                b'%' => Token::Modulo(Box::new(l), Box::new(r)),
                _ => Token::Exponent(Box::new(l), Box::new(r)),
            }
        })
}

// this is like value in json parser
fn expression<'a>() -> Parser<'a, u8, Token> {
    trailing_atomic_expr() | leading_atomic_expr() | operator() | variable() | number()
}

pub fn total_expr<'a>() -> Parser<'a, u8, Token> {
    expression() - end()
}
