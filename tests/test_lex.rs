use y::lexer::Lexer;

fn main() {
    let mut lexer = Lexer::new("\"unterminated");
    let tokens = lexer.tokenize();
    println!("{:?}", tokens);
}
