use y_lang_compiler::lexer::*;

fn main() {
    let mut lexer = Lexer::new("\"unterminated");
    let tokens = lexer.tokenize();
    println!("{:?}", tokens);
}
