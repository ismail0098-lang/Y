mod lexer;

fn main() {
    let mut lexer = lexer::Lexer::new("\"unterminated");
    let tokens = lexer.tokenize();
    println!("{:?}", tokens);
}
