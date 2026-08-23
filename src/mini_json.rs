//! A minimal JSON reader.
//!
//! This crate's `[dependencies]` are deliberately empty, and pulling in serde so
//! the compiler can read a verifying key and an input map would be a poor
//! trade. Only what those two files need is implemented: objects, arrays,
//! strings and numbers. Numbers are kept as their source text, never parsed
//! into a float - field elements routinely exceed 2^53 and `f64` would round
//! them silently.

/// A parsed JSON value. Only what a verifying key needs.
///
/// Hand-rolled because this crate's `[dependencies]` are deliberately empty -
/// pulling in serde so the compiler can read one small fixed-shape file would
/// be a poor trade.
#[derive(Debug, Clone)]
pub enum Json {
    Str(String),
    Num(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
    Other,
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
    /// A field element, however it was written: `"123"` or `123`.
    pub fn scalar(&self) -> Option<String> {
        match self {
            Json::Str(s) | Json::Num(s) => Some(s.clone()),
            _ => None,
        }
    }
}

pub struct P<'a> {
    pub b: &'a [u8],
    pub i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    pub fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            None => Err("unexpected end of JSON".into()),
            Some(b'{') => {
                self.i += 1;
                let mut out = Vec::new();
                loop {
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(out));
                        }
                        Some(b',') => {
                            self.i += 1;
                        }
                        Some(b'"') => {
                            let k = self.string()?;
                            self.ws();
                            if self.b.get(self.i) != Some(&b':') {
                                return Err(format!("expected ':' after key {:?}", k));
                            }
                            self.i += 1;
                            out.push((k, self.value()?));
                        }
                        other => return Err(format!("unexpected {:?} in object", other)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut out = Vec::new();
                loop {
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(out));
                        }
                        Some(b',') => self.i += 1,
                        None => return Err("unterminated array".into()),
                        _ => out.push(self.value()?),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(c) if c.is_ascii_digit() || *c == b'-' => {
                let start = self.i;
                while self
                    .b
                    .get(self.i)
                    .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E'))
                {
                    self.i += 1;
                }
                Ok(Json::Num(
                    String::from_utf8_lossy(&self.b[start..self.i]).into_owned(),
                ))
            }
            Some(_) => {
                // true / false / null - not needed, skip the bare word.
                while self.b.get(self.i).is_some_and(|c| c.is_ascii_alphabetic()) {
                    self.i += 1;
                }
                Ok(Json::Other)
            }
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err("expected string".into());
        }
        self.i += 1;
        let mut out = String::new();
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    out.push(match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                }
                _ => out.push(c as char),
            }
        }
        Err("unterminated string".into())
    }
}

