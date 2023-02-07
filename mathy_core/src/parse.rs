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
            Token::Var(var_name) => write!(f, "{}", var_name),
            _ => todo!("woops"),
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
                Token::Complex(c) => Token::Exponent(Box::new(expr1), Box::new(Token::Complex(c))),
                // TODO: refactor todos to a proper Error
                _ => todo!("nooo"),
            },
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

fn space<'a>() -> Parser<'a, u8, ()> {
    one_of(b" \t\r\n").repeat(0..).discard()
}

fn operator<'a>() -> Parser<'a, u8, Token> {
    let parser = number().opt() - space() + variable().opt() - space() + one_of(b"+-*/^%")
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
                    (Token::Complex(c1), Token::Complex(c2)) => Token::Exponent(
                        Box::new(Token::Complex(c1.clone())),
                        Box::new(Token::Complex(c2.clone())),
                    ),
                    (Token::Complex(c1), Token::Add(inner_l, inner_r)) => Token::Add(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Multiply(inner_l, inner_r)) => Token::Multiply(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (Token::Complex(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(
                        Box::new(Token::Exponent(
                            Box::new(Token::Complex(c1.clone())),
                            inner_l,
                        )),
                        inner_r,
                    ),
                    (l, r) => todo!("hi"),
                },
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
        Token::Complex(Complex::new(n, 0.0))
    }
}
