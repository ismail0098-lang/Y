#[cfg(test)]
mod tests {
    use crate::lexer::*;

    #[test]
    fn test_unterminated_string() {
        let mut lexer = Lexer::new("\"unterminated");
        let tokens = lexer.tokenize();
        println!("{:?}", tokens);
    }
}
