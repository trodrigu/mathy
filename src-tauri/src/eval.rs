use derivative::*;
use num::ToPrimitive;
use std::collections::HashMap;

use crate::parse::{Token, Type, Type::*};

#[derive(Derivative)]
#[derivative(PartialEq, Clone, Debug)]
pub enum Error {
    DivideByZero {
        expr: Token,
        numerator: Token,
        denominator: Token,
    },
}

impl Token {
    pub(crate) fn expr_type(&self) -> Type {
        match self {
            Token::Complex(n) => Number(n.clone()),
            Token::Add(_, _) => Arithmetic,
            Token::Subtract(_, _) => Arithmetic,
            Token::Multiply(_, _) => Arithmetic,
            Token::Divide(_, _) => Arithmetic,
            Token::Exponent(_, _) => Arithmetic,
            Token::Modulo(_, _) => Arithmetic,
            _ => Unknown,
        }
    }

    pub fn eval(&self, context: HashMap<String, Token>) -> Result<Self, Error> {
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

    fn eval_step_binary(
        &self,
        c1: &Self,
        c2: &Self,
        context: HashMap<String, Token>,
    ) -> Result<Self, Error> {
        let c1 = c1.eval_step(context.clone())?;
        let c2 = c2.eval_step(context.clone())?;

        match (self, c1.expr_type(), c2.expr_type()) {
            (Token::Add(_, _), Type::Number(inner_c1), Type::Number(inner_c2)) => {
                Ok(Token::Complex(inner_c1 + inner_c2))
            }
            (Token::Subtract(_, _), Type::Number(inner_c1), Type::Number(inner_c2)) => {
                Ok(Token::Complex(inner_c1 - inner_c2))
            }
            (Token::Multiply(_, _), Type::Number(inner_c1), Type::Number(inner_c2)) => {
                Ok(Token::Complex(inner_c1 * inner_c2))
            }
            (Token::Divide(_, _), Type::Number(inner_c1), Type::Number(inner_c2)) => {
                Ok(Token::Complex(inner_c1 / inner_c2))
            }
            (Token::Exponent(_, _), Type::Number(inner_c1), Type::Number(inner_c2)) => {
                Ok(Token::Complex(inner_c1.powf(inner_c2.to_f32().unwrap())))
            }
            _ => todo!("hi"),
        }
    }

    fn eval_step(&self, context: HashMap<String, Token>) -> Result<Self, Error> {
        let expr = &self;
        match expr {
            Token::Add(c1, c2)
            | Token::Subtract(c1, c2)
            | Token::Multiply(c1, c2)
            | Token::Divide(c1, c2)
            | Token::Exponent(c1, c2) => expr.eval_step_binary(c1, c2, context),
            Token::Complex(c) => Ok(Token::Complex(*c)),
            Token::Var(var) => {
                if let Some(v) = context.get(var) {
                    Ok(v.clone())
                } else {
                    panic!("no var!")
                }
            }
            _ => todo!("hi"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{total_expr, Complex, Token};

    #[test]
    fn test_eval_simple_int() {
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
    fn test_eval_simple_exponent() {
        e(b"2.0^2.0", real_num(4.0), None);
        e(b"(2.0*2.0)^2.0", real_num(16.0), None);
        e(b"2.0^(2.0*3.0)", real_num(64.0), None);
    }

    #[test]
    fn test_eval_simple_with_space() {
        e(b"3.0 *2.0", real_num(6.0), None);
        e(b"3.0* 2.0", real_num(6.0), None);
        e(b"3.0* ( 2.0+2.0)", real_num(12.0), None);
    }

    #[test]
    fn test_eval_vars() {
        let mut h = HashMap::new();
        h.insert("x".to_string(), real_num(2.0));
        h.insert("y".to_string(), real_num(2.0));
        e(b"2x+1", real_num(5.0), Some(h.clone()));
        e(b"y+1", real_num(3.0), Some(h.clone()));
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
    fn e(string: &'static [u8], expression: Token, context: Option<HashMap<String, Token>>) {
        let c = if let Some(h) = context {
            h
        } else {
            HashMap::new()
        };

        assert_eq!(total_expr().parse(string).unwrap().eval(c), Ok(expression));
    }

    fn real_num(n: f32) -> Token {
        Token::Complex(Complex::new(n, 0.0))
    }
}
