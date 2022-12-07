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

//fn parse_token(char: &str) -> Token {
    //match char {
        //"a" => Token::Var(String::from("a")),
        //"b" => Token::Var(String::from("b")),
        //"c" => Token::Var(String::from("c")),
        //"d" => Token::Var(String::from("d")),
        //"e" => Token::Var(String::from("e")),
        //"f" => Token::Var(String::from("f")),
        //"g" => Token::Var(String::from("g")),
        //"h" => Token::Var(String::from("h")),
        //"i" => Token::Var(String::from("i")),
        //"j" => Token::Var(String::from("j")),
        //"k" => Token::Var(String::from("k")),
        //"l" => Token::Var(String::from("l")),
        //"m" => Token::Var(String::from("m")),
        //"n" => Token::Var(String::from("n")),
        //"o" => Token::Var(String::from("o")),
        //"p" => Token::Var(String::from("p")),
        //"q" => Token::Var(String::from("q")),
        //"r" => Token::Var(String::from("r")),
        //"s" => Token::Var(String::from("s")),
        //"t" => Token::Var(String::from("t")),
        //"u" => Token::Var(String::from("u")),
        //"v" => Token::Var(String::from("v")),
        //"w" => Token::Var(String::from("w")),
        //"x" => Token::Var(String::from("x")),
        //"y" => Token::Var(String::from("y")),
        //"z" => Token::Var(String::from("z")),
        //"+" => Token::Plus,
        //"-" => Token::Minus,
        //"/" => Token::Division,
        //"*" => Token::Multiplication,
        //"=" => Token::Equal,
        //"^" => Token::Exponent,
        //_ => Token::Constant(char.parse::<f32>().unwrap()),
    //}
//}

//fn parse_char(char: &str) -> Option<Token> {
    //match char {
        //" " => None,
        //parsed_char => Some(parse_token(parsed_char)),
    //}
//}

fn parse_term(maybe_term: &str) -> Token {
    // -
    // could be constant or coefficient
    // 2
    // could be constant or coefficient
    // x
    // its a coefficient
    // sign numeric alphabetic
    // numeric alphabetic
    // sign numeric alphabetic ^ numeric
    // alphabetic ^ numeric
    // alphabetic
    // numeric alphabetic ^ numeric
    // sign numeric
    // numeric
    // ["-2x", "-2"]
    if is_equal(maybe_term.to_string()) {
        Token::Equal
    } else if is_operator(maybe_term.to_string()) {
        parse_operator(maybe_term)
    } else {
        maybe_term
            .split('^')
            .fold(Token::Constant(1.0), |acc, el| {
                el
                    .chars()
                    .fold(acc, |inner_acc, inner_el| {
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
        _ => Token::Constant(1.0)
    }
}

fn parse_equation(equation: &str) -> f32 {
    let mut parsed_equation: Vec<Token> = Vec::new();

    for maybe_term in equation.split(' ') {
        // -2x^2 + 2 = x + 1
        // [-2x^2, +, 2, = , x, 1]
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

    //let mut left_grouped: Vec<Term> = group_tokens(left);
    //let mut right_grouped: Vec<Term> = group_tokens(right);

    //dbg!(left_grouped.clone());
    //dbg!(right_grouped.clone());

    //let new_left: Vec<Term> = Vec::new();

    //for left_el in left_grouped.iter() {
        //if matches!(left_el, Token::Var) {
            //new_left.push(left_el);
        //} else {
            //right_grouped.push(left_el);
        //}
    //}

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
    //let (constants_and_operators_left, vars_left): (Vec<Vec<Token>>, Vec<Vec<Token>>) =
    //left_grouped
    //.into_iter()
    //.partition(|el| el.iter().any(|t| matches!(t, Token::Constant(_))));

    //let (constants_and_operators_right, vars_right): (Vec<Vec<Token>>, Vec<Vec<Token>>) =
    //right_grouped
    //.into_iter()
    //.partition(|el| el.iter().any(|t| matches!(t, Token::Constant(_))));

    //let mut constants_and_operators_left_tuples: Vec<(Option<Token>, Token)> =
    //tupelize_side(constants_and_operators_left);
    //let mut constants_and_operators_right_tuples: Vec<(Option<Token>, Token)> =
    //tupelize_side(constants_and_operators_right);
    //let mut vars_left_tuples: Vec<(Option<Token>, Token)> = tupelize_side(vars_left);
    //let mut vars_right_tuples: Vec<(Option<Token>, Token)> = tupelize_side(vars_right);

    //dbg!(constants_and_operators_left_tuples.clone());
    //dbg!(constants_and_operators_right_tuples.clone());
    //dbg!(vars_left_tuples.clone());
    //dbg!(vars_right_tuples.clone());

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

type MaybeSign = Option<Token>;
type MaybeCoefficientOrConstant = Option<Token>;
type MaybeVariable = Option<Token>;
type MaybeExponent = Option<Token>;
type MaybeEquationOperator = Option<Token>;
type Term = (
    MaybeSign,
    MaybeCoefficientOrConstant,
    MaybeVariable,
    MaybeExponent,
    MaybeEquationOperator,
);

fn default_term() -> Term {
    (None, None, None, None, None)
}

//fn group_tokens(tokens: Vec<Token>) -> Vec<Term> {
    //let last_index = tokens.len() - 1;
    //// first/last rules
    //// equation operator can never be first or last
    //// term operator or sign can be first but never last
    //// constant can be last
    //// var can be last
    //let (finished_terms, _on_deck_term, _exponent_prev) = tokens.iter().enumerate().fold(
        //(Vec::new(), default_term(), false),
        //|acc: (Vec<Term>, Term, bool), (i, el)| {
            ////acc: (finished_terms: Vec<Term>, on_deck_term: Term, exponent_prev: bool)
            //let mut acc_cloned = acc.clone();
            //let (mut finished_terms, mut on_deck_term, mut exponent_prev) = acc_cloned;
            //let (sign, coefficient_or_constant, variable, exponent, equation_operator) =
                //on_deck_term;
            //// how to know when a term is full?
            //// pipe into the first one if full
            //// otherwise build term
            //// ([], on deck)?
            //// if finishing operator (constant or var)
            //// if constant check that after is an equation operator
            //// 2 x +
            //// can we do access from inside loop?
            //// 2 -
            //// set flag that previous was an exponent operator?
            //match el {
                //Token::Exponent => (
                    //finished_terms,
                    //(
                        //sign,
                        //coefficient_or_constant,
                        //variable,
                        //exponent,
                        //equation_operator,
                    //),
                    //true,
                //),
                //Token::Var(var) => {
                    //// pop and push as constant is assumed the ender!
                    //if let Some(last_finished_term) = finished_terms.pop() {
                        //let updated_on_deck_term = (
                            //sign,
                            //coefficient_or_constant,
                            //Some(Token::Var(var.to_string())),
                            //exponent,
                            //equation_operator,
                        //);
                        //finished_terms.push(updated_on_deck_term);
                        //(finished_terms, (None, None, None, None, None), false)
                    //} else {
                        //let updated_on_deck_term = (
                            //sign,
                            //coefficient_or_constant,
                            //Some(Token::Var(var.to_string())),
                            //exponent,
                            //equation_operator,
                        //);
                        //finished_terms.push(updated_on_deck_term);
                        //(finished_terms, (None, None, None, None, None), false)
                    //}
                //}
                //Token::Constant(val) => {
                    //if exponent_prev {
                        //let updated_on_deck_term = (
                            //sign,
                            //Some(Token::Constant(*val)),
                            //variable,
                            //exponent,
                            //equation_operator,
                        //);
                        //finished_terms.push(updated_on_deck_term);
                        //(finished_terms, (None, None, None, None, None), false)
                    //} else {
                        //let updated_on_deck_term = (
                            //sign,
                            //Some(Token::Constant(*val)),
                            //variable,
                            //exponent,
                            //equation_operator,
                        //);
                        //finished_terms.push(updated_on_deck_term);
                        //(finished_terms, (None, None, None, None, None), false)
                    //}
                //}
                //Token::Plus => {
                    //// ++ situation move from equation operator to sign
                    //// i == 1 then sign
                    //update_on_deck_term_sign_or_equation_operator(
                        //Token::Plus,
                        //finished_terms,
                        //(
                            //sign,
                            //coefficient_or_constant,
                            //variable,
                            //exponent,
                            //equation_operator,
                        //),
                        //exponent_prev,
                        //i,
                    //)
                //}
                //Token::Minus => update_on_deck_term_sign_or_equation_operator(
                    //Token::Minus,
                    //finished_terms,
                    //(
                        //sign,
                        //coefficient_or_constant,
                        //variable,
                        //exponent,
                        //equation_operator,
                    //),
                    //exponent_prev,
                    //i,
                //),
                //Token::Multiplication => {
                    //add_equation_operator(Token::Multiplication, finished_terms, i)
                //}
                //Token::Division => add_equation_operator(Token::Division, finished_terms, i),
                //_ => acc,
            //}
        //},
    //);

    //finished_terms
//}

fn add_equation_operator(
    equation_token: Token,
    mut finished_terms: Vec<Term>,
    index: usize,
) -> (Vec<Term>, Term, bool) {
    assert!(index != 0);

    let updated_on_deck_term = (None, None, None, None, Some(equation_token));
    finished_terms.push(updated_on_deck_term);
    (finished_terms, (None, None, None, None, None), false)
}

fn update_on_deck_term_sign_or_equation_operator(
    sign_token: Token,
    finished_terms: Vec<Term>,
    on_deck_term: Term,
    exponent_prev: bool,
    index: usize,
) -> (Vec<Term>, Term, bool) {
    let (sign, coefficient_or_constant, variable, exponent, equation_operator) = on_deck_term;

    if index == 0 {
        let updated_on_deck_term = (
            Some(sign_token),
            coefficient_or_constant,
            variable,
            exponent,
            equation_operator,
        );
        (finished_terms, updated_on_deck_term, false)
    } else {
        if let Some(equation_operator_token) = equation_operator {
            let updated_on_deck_term = (
                Some(sign_token),
                coefficient_or_constant,
                variable,
                exponent,
                Some(equation_operator_token),
            );
            (finished_terms, updated_on_deck_term, false)
        } else {
            add_equation_operator(Token::Plus, finished_terms, index)
        }
    }
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
