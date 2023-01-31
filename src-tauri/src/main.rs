#![feature(iter_next_chunk)]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use derivative::*;
use pom::parser::Parser;
use pom::parser::*;
use std::collections::HashMap;
use std::rc::Rc;
use num::{Signed, Zero};
use std::fmt::{Display, Formatter, Result as Res};

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn evaluate_equation(equation: &str) -> String {
    let result: Result<Token, pom::Error> = expression().parse(equation.as_bytes());
    match result {
        //Ok(token) => eval(token, 2.0).to_string(),
        Ok(token) => "hi".to_string(),
        Err(_err) => {
            panic!("nope")
        }
    }
}

#[derive(Derivative)]
#[derivative(PartialEq, Clone, Debug)]
pub enum Error {
    DivideByZero {
        expr: Token,
        numerator: Token,
        denominator: Token,
    },
}

//pub type Function = dyn Fn(&Token, &[Token], &HashMap<String, Token>) -> Result<Token, Error>;

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
                    write!(f, "{} {} {}i", z.re.clone(), if z.im.is_negative() { "-" } else { "+" }, z.im.abs())
                }
            },
            _ => todo!("hi"),
        }
    }
}

pub type Integer = num::bigint::BigInt;
//pub type Rational = num::rational::Ratio<Integer>;
pub type Complex = num::complex::Complex<f32>;

#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Type {
    Number(Complex),
    Arithmetic,
    Unknown,
}

impl Token {

    pub(crate) fn expr_type(&self) -> Type {
        use crate::Type::*;

        match self {
            Token::Complex(n) => Number(n.clone()),
            Token::Add(_,_) => Arithmetic,
            Token::Subtract(_,_) => Arithmetic,
            Token::Multiply(_,_) => Arithmetic,
            Token::Divide(_,_) => Arithmetic,
            Token::Exponent(_,_) => Arithmetic,
            Token::Modulo(_,_) => Arithmetic,
            _ => Unknown
        }
    }

    fn eval(&self, context: HashMap<String, Token>) -> Result<Self, Error> {
        let mut old_expr: Self = self.clone();

        loop {
            let new_context = context.clone();

            let new_expr: Self = old_expr.eval_step(new_context)?;

            if new_expr == old_expr {
                return Ok(new_expr);
            }

            old_expr = new_expr;
        }
    }

    fn eval_step_binary(&self, c1: &Self, c2: &Self, context: HashMap<String, Token>) -> Result<Self, Error> {
        let c1 = c1.eval_step(context.clone())?;
        let c2 = c2.eval_step(context.clone())?;

        dbg!(self.clone());
        dbg!(c1.clone());
        dbg!(c2.clone());

        match (self, c1.expr_type(), c2.expr_type()) {
            (Token::Add(_,_), Type::Number(inner_c1), Type::Number(inner_c2)) => Ok(Token::Complex(inner_c1 + inner_c2)),
            (Token::Subtract(_,_), Type::Number(inner_c1), Type::Number(inner_c2)) => Ok(Token::Complex(inner_c1 - inner_c2)),
            (Token::Multiply(_,_), Type::Number(inner_c1), Type::Number(inner_c2)) => Ok(Token::Complex(inner_c1 * inner_c2)),
            (Token::Divide(_,_), Type::Number(inner_c1), Type::Number(inner_c2)) => Ok(Token::Complex(inner_c1 / inner_c2)),
            _ => todo!("hi"),
        }
    }

    fn eval_step(&self, context: HashMap<String, Token>) -> Result<Self, Error> {
        let expr = &self;
        match expr {
            Token::Add(c1, c2) | Token::Subtract(c1, c2) | Token::Multiply(c1, c2) | Token::Divide(c1, c2) => expr.eval_step_binary(c1, c2, context),
            Token::Complex(c) => Ok(Token::Complex(*c)),
            Token::Var(var) => if let Some(v) = context.get(var) { Ok(v.clone()) } else { panic!("no var!") },
            t => {
                dbg!(t.clone());
                todo!("hi")},
        }
    }


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
    let p = number()  + one_of(b"+-*/^%") + sym(b'(') + call(expression) + sym(b')');
    p.name("trailing_atomic_expr").map(|((((expr1, op), _left_paren), expr2), _right_paren)| {
        match op {
            b'*' => {
                match expr2 {
                    Token::Divide(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    },
                    Token::Add(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Add(l, r)))
                    },
                    Token::Subtract(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    },
                    _ => todo!("nooo"),
                }
                
            },
            b'/' => {
                match expr2 {
                    Token::Multiply(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Multiply(l, r)))
                    },
                    Token::Add(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Add(l, r)))
                    },
                    Token::Subtract(l, r) => {
                        Token::Divide(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    },
                    _ => todo!("nooo"),
                }
                
            },
            b'+' => {
                match expr2 {
                    Token::Multiply(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Multiply(l, r)))
                    },
                    Token::Divide(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Divide(l, r)))
                    },
                    Token::Subtract(l, r) => {
                        Token::Add(Box::new(expr1), Box::new(Token::Subtract(l, r)))
                    },
                    _ => todo!("nooo"),
                }
                
            },
            t => {
                dbg!(std::str::from_utf8(&[t]).clone());
                todo!("noooo")
            },
        }
    })
}
fn leading_atomic_expr<'a>() -> Parser<'a, u8, Token> {
    let p = sym(b'(') + call(expression) + sym(b')') + one_of(b"+-*/^%") + call(expression);
    p.name("leading_atomic_expr").map(|((((_left_paren, expr1), _right_paren), op), expr2)| {
        match op {
            b'/' => {
                match expr2 {
                    Token::Complex(c) => Token::Divide(Box::new(expr1), Box::new(Token::Complex(c))),
                    _ => todo!("nooo"),
                }
                
            },
            b'*' => {
                match expr2 {
                    Token::Complex(c) => Token::Multiply(Box::new(expr1), Box::new(Token::Complex(c))),
                    _ => todo!("nooo"),
                }
                
            },
            b'+' => {
                match expr2 {
                    Token::Complex(c) => Token::Add(Box::new(expr1), Box::new(Token::Complex(c))),
                    _ => todo!("nooo"),
                }
                
            },
            t => {
                //dbg!(t.clone());
                dbg!(std::str::from_utf8(&[t]).clone());

                todo!("noooo")
            },
        }
    })
}

fn operator<'a>() -> Parser<'a, u8, Token> {
    let parser = number() + variable().opt() + one_of(b"+-*/^%") + call(expression);
    parser.name("regular_operator").map(|(((mut l, v), op), r)| {
        //dbg!(l.clone());
        //dbg!(std::str::from_utf8(&[op]).clone());
        //dbg!(r.clone());
        if let Some(var) = v {
            l = Token::Multiply(Box::new(l), Box::new(var));
        }

        match op {
            b'+' => Token::Add(Box::new(l), Box::new(r)),
            b'-' => Token::Subtract(Box::new(l), Box::new(r)),
            b'*' => {
                match (l,r) {
                    (Token::Complex(c1), Token::Complex(c2)) => Token::Multiply(Box::new(Token::Complex(c1.clone())), Box::new(Token::Complex(c2.clone()))),
                    (Token::Complex(c1), Token::Add(inner_l, inner_r)) => Token::Add(Box::new(Token::Multiply(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (Token::Complex(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(Box::new(Token::Multiply(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (Token::Complex(c1), Token::Multiply(inner_l, inner_r)) => Token::Multiply(Box::new(Token::Multiply(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (l,r) => todo!("hi"),
                }
                
            },
            b'/' => {
                match (l,r) {
                    (Token::Complex(c1), Token::Complex(c2)) => Token::Divide(Box::new(Token::Complex(c1.clone())), Box::new(Token::Complex(c2.clone()))),
                    (Token::Complex(c1), Token::Add(inner_l, inner_r)) => Token::Add(Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (Token::Complex(c1), Token::Subtract(inner_l, inner_r)) => Token::Subtract(Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (Token::Complex(c1), Token::Divide(inner_l, inner_r)) => Token::Divide(Box::new(Token::Divide(Box::new(Token::Complex(c1.clone())), inner_l)), inner_r),
                    (l,r) => todo!("hi"),
                }
                
            },
            b'^' => Token::Exponent(Box::new(l), Box::new(r)),
            b'%' => Token::Modulo(Box::new(l), Box::new(r)),
            _ => Token::Exponent(Box::new(l), Box::new(r)),
        }
    })
}

// this is like value in json parser
fn expression<'a>() -> Parser<'a, u8, Token> {
    trailing_atomic_expr() | leading_atomic_expr() | operator()  | variable() | number()
}

fn total_expr<'a>() -> Parser<'a, u8, Token> {
    expression() - end()
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
    //use num::rational::{Ratio, BigRational};

    #[test]
    fn test_parse_simple_float() {
        t(b"2.0+3.0-4.0", Token::Add(
            Box::new(real_num(2.0)),
            Box::new(Token::Subtract(
                Box::new(real_num(3.0)),
                Box::new(real_num(4.0)),
            )),
        ));
    }

    #[test]
    fn test_parse_simple_int() {
        e(b"2+3-4", real_num(1.0), None);
    }

    #[test]
    fn test_eval_simple_float() {
        e(b"2.0+3.0-4.0", real_num(1.0), None);
    }

    #[test]
    fn test_eval_simple_float_mult() {
        e(b"2.0+3.0*4.0", real_num(14.0), None);
        e(b"2.0*3.0+4.0", real_num(10.0), None);
        e(b"2.0*3.0-4.0", real_num(2.0), None);
        e(b"2.0*3.0-4.0*3.0", real_num(-6.0), None);
        e(b"2.0*3.0-4.0*3.0*2.0", real_num(-18.0), None);
        e(b"2.0*3.0-4.0*3.0*2.0*2.0", real_num(-42.0), None);
        e(b"2.0*3.0*2.0-4.0*3.0*2.0*2.0", real_num(-36.0), None);
    }

    #[test]
    fn test_eval_simple_float_div() {
        e(b"2.0+3.0/4.0", real_num(2.75), None);
        e(b"2.0/3.0+4.0", real_num(4.6666665), None);
        e(b"2.0/3.0-4.0", real_num(-3.3333333), None);
        e(b"2.0/3.0-4.0/3.0", real_num(-0.6666667), None);
        e(b"2.0/3.0-4.0/3.0/2.0", real_num(0.0), None);
        e(b"2.0/3.0-4.0/3.0/2.0/2.0", real_num(-0.6666667), None);
        e(b"2.0/3.0/2.0-4.0/3.0/2.0/2.0", real_num(-0.0), None);
    }

    #[test]
    fn test_eval_simple_float_parens_div() {
        e(b"(2.0+3.0)/4.0", real_num(1.25), None);
        e(b"(2.0+3.0+3.0)/4.0", real_num(2.00), None);
        e(b"(3.0+(3.0/3.0))/4.0", real_num(1.00), None);

        e(b"4.0/(2.0+3.0)", real_num(0.80), None);
        e(b"4.0/(2.0+3.0+3.0)", real_num(0.50), None);
        e(b"4.0/((2.0*3.0)+2.0)", real_num(0.50), None);
    }

    #[test]
    fn test_eval_vars() {
        let mut h = HashMap::new();
        h.insert("x".to_string(), real_num(2.0));
        e(b"2x+1", real_num(5.0), Some(h));
    }

    //#[test]
    //fn test_parse_define_polynomial_with_double_neg() {
    //}

    //#[test]
    //}

    //#[test]
    //fn test_simplify_double_neg_with_parens() {
    //}

    //#[test]
    //fn test_simplify_double_pos_with_parens() {
    //}

    //#[test]
    //fn test_eval_exponent_polynomial() {
    //}

    //#[test]
    //fn test_eval_exponent_polynomial_with_variable_sub() {
    //}

    //#[test]
    //fn test_eval_multiply() {
    //}

    //#[test]
    //fn test_eval_multiply_parens() {
    //}

    //#[test]
    //fn test_eval_division() {
    //}

    //#[test]
    //fn test_eval_addition() {
    //}

    //#[test]
    //fn test_eval_subtract() {
    //}

    //#[test]
    //fn test_eval_polynomial_double_neg() {
    //}

    //#[test]
    //fn test_eval_fraction_parens_polynomial() {
    //}

    #[track_caller]
    fn t(string: &'static [u8], expression: Token) {
        dbg!(std::str::from_utf8(&[120]).clone());
        let res = total_expr().parse(string);
        dbg!(res.clone());
        assert_eq!(res, Ok(expression));
    }

    #[track_caller]
    fn e(string: &'static [u8], expression: Token, mut context: Option<HashMap<String, Token>>) {
        dbg!(std::str::from_utf8(&[120]).clone());
        let res = total_expr().parse(string);
        dbg!(res.clone());

        let c = if let Some(h) = context { h } else { HashMap::new() };

        assert_eq!(total_expr().parse(string).unwrap().eval(c), Ok(expression));
    }

    fn real_num(n: f32) -> Token {
        Token::Complex(Complex::new(n, 0.0))
    }
}
