$f = 'parser.rs'
$txt = [IO.File]::ReadAllText($f, [Text.Encoding]::UTF8)

$old = '            TokenKind::U64 => { self.advance(); return Ok(Type::Primitive("U64".into(), span)); }'
$new = '            TokenKind::U64 => { self.advance(); return Ok(Type::Primitive("U64".into(), span)); }' + "`r`n            TokenKind::QFixed(s) => { self.advance(); return Ok(Type::Primitive(s.clone(), span)); }"

if ($txt.Contains($old)) {
    $txt = $txt.Replace($old, $new)
    [IO.File]::WriteAllText($f, $txt, [Text.Encoding]::UTF8)
    Write-Host 'parser.rs updated OK'
} else {
    Write-Host 'TARGET NOT FOUND'
}
